const std = @import("std");
const model_catalog = @import("../core/gateway/model_catalog.zig");
const gateway_provider = @import("../core/gateway/gateway_provider.zig");
const io_mod = @import("../core/shared/io.zig");
const secret = @import("../core/auth/secret.zig");
const sort_utils = @import("../core/shared/sort_utils.zig");
const local = @import("local.zig");
const gateway_client = @import("client.zig");

/// Reads the server's `GET /v1/models` listing. A local server exposes only
/// what the operator loaded, so the listing needs no tool-support or pricing
/// filter: every listed model runs on this machine.
const max_catalog_models: usize = 4096;
const max_model_id_bytes: usize = 256;
const max_catalog_bytes: usize = 4 * 1024 * 1024;
const fetch_timeout_ms: i64 = 30_000;

pub const model_catalog_provider = model_catalog.Provider{
    .fetch_fn = fetchCatalogForProvider,
};

pub const cli_model_catalog_provider = gateway_provider.CliModelCatalogProvider{
    .fetch_fn = fetchCliModelCatalog,
};

fn fetchCliModelCatalog(
    _: ?*anyopaque,
    alloc: std.mem.Allocator,
    input: gateway_provider.CliModelCatalogInput,
) gateway_provider.CliModelCatalogResult {
    return switch (model_catalog.fetchWithPublicFallback(model_catalog_provider, alloc, .{
        .access = input.access,
        .endpoint = input.endpoint,
        .cancel_flag = input.cancel_flag,
        .view = .full,
    })) {
        .loaded => |loaded| blk: {
            var catalog = loaded.catalog;
            defer model_catalog.freeModelCatalog(alloc, &catalog);
            const ids = model_catalog.projectModelIds(alloc, catalog.items) catch return .{ .failure = .{
                .access = loaded.provenance.access,
                .anonymous_fallback_used = false,
                .failure = .{ .category = .resource_exhausted },
            } };
            break :blk .{ .loaded = .{ .ids = ids, .provenance = loaded.provenance } };
        },
        .failed => |failure| .{ .failure = failure },
    };
}

fn fetchCatalogForProvider(
    _: ?*anyopaque,
    alloc: std.mem.Allocator,
    input: model_catalog.FetchInput,
) std.mem.Allocator.Error!model_catalog.ProviderResult {
    const request_url = local.resolveModelsEndpoint(alloc) catch |err| {
        if (err == error.OutOfMemory) return error.OutOfMemory;
        return .{ .failure = .{ .category = .runtime } };
    };
    defer alloc.free(request_url);

    var fallback_cancel = std.atomic.Value(bool).init(false);
    const cancel_flag = input.cancel_flag orelse &fallback_cancel;
    const deadline = std.Io.Clock.Timestamp.fromNow(io_mod.getIo(), .{
        .clock = .awake,
        .raw = .fromMilliseconds(fetch_timeout_ms),
    });

    // Most local servers ignore authorization; send the configured key when
    // one exists so gated proxies (vLLM api-key mode) accept the request.
    var response = fetchCatalogResponse(
        alloc,
        request_url,
        input.access.authorizationCredential(),
        cancel_flag,
        deadline,
    ) catch |err| {
        if (err == error.OutOfMemory) return error.OutOfMemory;
        return .{ .failure = catalogFetchFailure(err) };
    };
    defer response.deinit(alloc);
    if (response.status != .ok) {
        return .{ .failure = model_catalog.failureForHttpStatus(response.status) };
    }

    const catalog = parseCatalog(alloc, response.body) catch |err| {
        if (err == error.OutOfMemory) return error.OutOfMemory;
        return .{ .failure = .{ .category = .malformed_response, .http_status = .ok } };
    };
    return .{ .catalog = catalog };
}

fn catalogFetchFailure(err: anyerror) model_catalog.Failure {
    if (err == error.Cancelled) return .{ .category = .cancellation };
    if (err == error.LocalModelCatalogTooLarge) return .{ .category = .malformed_response };
    return .{ .category = .transport, .retryable = true };
}

const FetchResponse = struct {
    status: std.http.Status,
    body: []u8,

    pub fn deinit(self: *FetchResponse, alloc: std.mem.Allocator) void {
        secret.zeroAndFree(alloc, self.body);
        self.* = undefined;
    }
};

const FetchOperation = struct {
    alloc: std.mem.Allocator,
    url: []const u8,
    credential: ?[]const u8,

    pub fn run(self: *@This()) !FetchResponse {
        var client: std.http.Client = .{ .allocator = self.alloc, .io = io_mod.getIo() };
        defer client.deinit();

        const auth_header = if (self.credential) |value|
            try std.fmt.allocPrint(self.alloc, "Bearer {s}", .{value})
        else
            null;
        defer if (auth_header) |header| secret.zeroAndFree(self.alloc, header);

        const body_buffer = try self.alloc.alloc(u8, max_catalog_bytes + 1);
        defer secret.zeroAndFree(self.alloc, body_buffer);
        var response_writer = std.Io.Writer.fixed(body_buffer);
        var extra_headers: [1]std.http.Header = .{.{ .name = "accept", .value = "application/json" }};

        const result = client.fetch(.{
            .location = .{ .url = self.url },
            .method = .GET,
            .headers = .{
                .authorization = if (auth_header) |header|
                    .{ .override = header }
                else
                    .omit,
                .user_agent = .{ .override = gateway_client.user_agent },
                .accept_encoding = .omit,
            },
            .extra_headers = &extra_headers,
            .response_writer = &response_writer,
            .redirect_behavior = .unhandled,
        }) catch |err| switch (err) {
            error.WriteFailed => return error.LocalModelCatalogTooLarge,
            else => return err,
        };
        const body = response_writer.buffered();
        if (body.len > max_catalog_bytes) return error.LocalModelCatalogTooLarge;
        return .{ .status = result.status, .body = try self.alloc.dupe(u8, body) };
    }
};

fn fetchCatalogResponse(
    alloc: std.mem.Allocator,
    url: []const u8,
    credential: ?[]const u8,
    cancel_flag: *std.atomic.Value(bool),
    deadline: std.Io.Clock.Timestamp,
) !FetchResponse {
    var operation = FetchOperation{ .alloc = alloc, .url = url, .credential = credential };
    return gateway_client.runBoundedHttpOperation(
        FetchResponse,
        alloc,
        cancel_flag,
        deadline,
        &operation,
    );
}

/// Parses an OpenAI `/models` listing (`{"data":[{"id":...,"created":...}]}`).
/// Fields beyond `id` and `created` are optional: llama.cpp, vLLM, MLX, and
/// Ollama's OpenAI shim each publish different subsets.
pub fn parseCatalog(
    alloc: std.mem.Allocator,
    catalog_json: []const u8,
) !std.ArrayList(model_catalog.ModelCatalogEntry) {
    var parsed = std.json.parseFromSlice(std.json.Value, alloc, catalog_json, .{}) catch
        return error.InvalidLocalModelCatalog;
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidLocalModelCatalog;
    const data = parsed.value.object.get("data") orelse return error.InvalidLocalModelCatalog;
    if (data != .array) return error.InvalidLocalModelCatalog;
    if (data.array.items.len > max_catalog_models) return error.InvalidLocalModelCatalog;

    var catalog: std.ArrayList(model_catalog.ModelCatalogEntry) = .empty;
    errdefer model_catalog.freeModelCatalog(alloc, &catalog);

    for (data.array.items) |entry| {
        if (entry != .object) continue;
        const object = entry.object;
        const id_value = object.get("id") orelse continue;
        if (id_value != .string) continue;
        if (id_value.string.len == 0 or id_value.string.len > max_model_id_bytes) continue;

        // A loaded local model can drive an agent turn; assume both tool use
        // and vision, which is true of every current agent-tuned checkpoint.
        try catalog.append(alloc, .{
            .id = try alloc.dupe(u8, id_value.string),
            .model_type = try alloc.dupe(u8, "language"),
            .released = if (object.get("created")) |value|
                if (value == .integer) value.integer else 0
            else
                0,
            .has_tool_use = true,
            .has_reasoning = false,
            .has_vision = true,
        });
    }

    sort_utils.sort(model_catalog.ModelCatalogEntry, catalog.items, {}, compareById);
    return catalog;
}

fn compareById(
    _: void,
    a: model_catalog.ModelCatalogEntry,
    b: model_catalog.ModelCatalogEntry,
) bool {
    return std.mem.lessThan(u8, a.id, b.id);
}

test "parses a llama.cpp style models listing" {
    const body =
        \\{"object":"list","data":[
        \\{"id":"qwen3.6-27b","object":"model","created":1788022026,"owned_by":"mlx"},
        \\{"id":"gpt-oss-20b","object":"model","owned_by":"llama.cpp"}
        \\]}
    ;
    var catalog = try parseCatalog(std.testing.allocator, body);
    defer model_catalog.freeModelCatalog(std.testing.allocator, &catalog);

    try std.testing.expectEqual(@as(usize, 2), catalog.items.len);
    try std.testing.expectEqualStrings("gpt-oss-20b", catalog.items[0].id);
    try std.testing.expectEqualStrings("qwen3.6-27b", catalog.items[1].id);
    try std.testing.expectEqual(@as(i64, 1788022026), catalog.items[1].released);
    try std.testing.expect(catalog.items[0].has_tool_use);
}

test "rejects malformed listings" {
    for ([_][]const u8{ "[]", "{}", "{\"data\":{}}" }) |body| {
        try std.testing.expectError(error.InvalidLocalModelCatalog, parseCatalog(std.testing.allocator, body));
    }
}

test "skips entries without a usable id" {
    const body =
        \\{"data":[{"object":"model"},{"id":"","object":"model"},{"id":"good","object":"model"}]}
    ;
    var catalog = try parseCatalog(std.testing.allocator, body);
    defer model_catalog.freeModelCatalog(std.testing.allocator, &catalog);
    try std.testing.expectEqual(@as(usize, 1), catalog.items.len);
}
