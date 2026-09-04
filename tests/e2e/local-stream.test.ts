import { describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { runFx } from "../evals/eval-helpers";

const MODEL = "qwen3.6-27b";

/// Mirrors a local server's `GET /v1/models`: whatever is loaded, in arbitrary
/// order, with optional fields missing the way llama.cpp and MLX omit them.
const CATALOG = {
  object: "list",
  data: [
    { id: MODEL, object: "model", created: 1788022026, owned_by: "mlx" },
    { id: "gpt-oss-20b", object: "model", owned_by: "llama.cpp" },
  ],
};

function sse(lines: string[]): Response {
  return new Response(lines.join("\n\n") + "\n\n", {
    headers: { "content-type": "text/event-stream" },
  });
}

type FakeMode = "text" | "tool" | "throttle";

export interface FakeLocalServer {
  chatRequests: string[];
  stop: () => void;
  env: Record<string, string>;
}

function startFakeLocalServer(mode: FakeMode = "text"): FakeLocalServer {
  const chatRequests: string[] = [];
  let chatCalls = 0;
  const server = Bun.serve({
    port: 0,
    hostname: "127.0.0.1",
    async fetch(request) {
      const url = new URL(request.url);
      if (url.pathname === "/v1/models") return Response.json(CATALOG);
      if (url.pathname !== "/v1/chat/completions") {
        return new Response("not found", { status: 404 });
      }
      chatRequests.push(await request.text());
      chatCalls += 1;

      // Throttle once, then succeed: fx retries rate limits with backoff.
      if (mode === "throttle" && chatCalls === 1) {
        return Response.json(
          { error: { code: 429, message: "Server busy" } },
          { status: 429, headers: { "retry-after": "1" } },
        );
      }
      if (mode === "tool" && chatCalls === 1) {
        return sse([
          `data: {"id":"gen-1","model":"${MODEL}","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"glob_files","arguments":"{\\"pat"}}]}}]}`,
          `data: {"id":"gen-1","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"tern\\":\\"*.txt\\",\\"path\\":\\".\\"}"}}]}}]}`,
          `data: {"id":"gen-1","choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":20,"completion_tokens":8}}`,
          "data: [DONE]",
        ]);
      }
      return sse([
        `data: {"id":"gen-2","model":"${MODEL}","choices":[{"delta":{"content":"LOCAL_"}}]}`,
        `data: {"id":"gen-2","choices":[{"delta":{"content":"DIRECT_RESPONSE"}}]}`,
        `data: {"id":"gen-2","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":9,"completion_tokens":4}}`,
        "data: [DONE]",
      ]);
    },
  });
  return {
    chatRequests,
    stop: () => server.stop(true),
    env: {
      LOCAL_API_KEY: "e2e-local-key",
      FX_LOCAL_BASE_URL: `http://127.0.0.1:${server.port}/v1`,
      FX_DISABLE_KEYCHAIN: "1",
    },
  };
}

function workspace(): { dir: string; cleanup: () => void } {
  const dir = mkdtempSync(join(tmpdir(), "fx-local-"));
  writeFileSync(join(dir, "alpha.txt"), "alpha\n");
  writeFileSync(join(dir, "beta.txt"), "beta\n");
  return { dir, cleanup: () => rmSync(dir, { recursive: true, force: true }) };
}

function envFor(fake: FakeLocalServer, home: string): Record<string, string> {
  return { ...fake.env, HOME: home, FX_MODEL: MODEL };
}

describe("local provider", () => {
  test("lists the models the server has loaded", async () => {
    const fake = startFakeLocalServer();
    const ws = workspace();
    try {
      const selected = await runFx(["provider", "local"], {
        cwd: ws.dir,
        env: envFor(fake, ws.dir),
      });
      expect(selected.stdout).toContain("Provider set to Local.");

      const listed = await runFx(["models"], { cwd: ws.dir, env: envFor(fake, ws.dir) });
      expect(listed.stdout).toContain(MODEL);
      expect(listed.stdout).toContain("gpt-oss-20b");
    } finally {
      ws.cleanup();
      fake.stop();
    }
  });

  test("streams a completion from the base-url endpoint", async () => {
    const fake = startFakeLocalServer("text");
    const ws = workspace();
    try {
      await runFx(["provider", "local"], { cwd: ws.dir, env: envFor(fake, ws.dir) });
      const asked = await runFx(["ask", "--no-save", "say hi"], {
        cwd: ws.dir,
        env: envFor(fake, ws.dir),
      });
      expect(asked.stdout).toContain("LOCAL_DIRECT_RESPONSE");

      const sent = JSON.parse(fake.chatRequests.at(-1)!);
      expect(sent.model).toBe(MODEL);
      expect(sent.stream).toBe(true);
      expect(sent.usage).toEqual({ include: true });
      expect(sent.messages.at(-1)).toEqual({ role: "user", content: "say hi" });
      expect(sent.tools[0].type).toBe("function");

      // Strict local servers (llama.cpp, MLX) reject a second system message.
      const roles: string[] = sent.messages.map((m: { role: string }) => m.role);
      expect(roles[0]).toBe("system");
      expect(roles.slice(1).indexOf("system")).toBe(-1);
    } finally {
      ws.cleanup();
      fake.stop();
    }
  });

  test("round-trips a streamed tool call", async () => {
    const fake = startFakeLocalServer("tool");
    const ws = workspace();
    try {
      await runFx(["provider", "local"], { cwd: ws.dir, env: envFor(fake, ws.dir) });
      const asked = await runFx(["ask", "--no-save", "--yolo", "list the files here"], {
        cwd: ws.dir,
        env: envFor(fake, ws.dir),
      });
      expect(asked.stdout).toContain("LOCAL_DIRECT_RESPONSE");

      // The replayed turn must carry the assistant tool call and its result in
      // Chat Completions shape.
      const replayed = JSON.parse(fake.chatRequests.at(-1)!);
      const assistant = replayed.messages.find(
        (m: { role: string; tool_calls?: unknown }) => m.role === "assistant" && m.tool_calls,
      );
      expect(assistant.tool_calls[0]).toMatchObject({
        id: "call_1",
        type: "function",
        function: { name: "glob_files", arguments: '{"pattern":"*.txt","path":"."}' },
      });
      const result = replayed.messages.find((m: { role: string }) => m.role === "tool");
      expect(result.tool_call_id).toBe("call_1");
      expect(result.content).toContain("alpha.txt");
    } finally {
      ws.cleanup();
      fake.stop();
    }
  });

  test("retries a throttled request and succeeds", async () => {
    const fake = startFakeLocalServer("throttle");
    const ws = workspace();
    try {
      await runFx(["provider", "local"], { cwd: ws.dir, env: envFor(fake, ws.dir) });
      const asked = await runFx(["ask", "--no-save", "say hi"], {
        cwd: ws.dir,
        env: envFor(fake, ws.dir),
      });
      expect(asked.stdout).toContain("LOCAL_DIRECT_RESPONSE");
      expect(fake.chatRequests.length).toBe(2);
    } finally {
      ws.cleanup();
      fake.stop();
    }
  });

  test("refuses a plain-HTTP base URL off loopback", async () => {
    const fake = startFakeLocalServer();
    const ws = workspace();
    try {
      await runFx(["provider", "local"], { cwd: ws.dir, env: envFor(fake, ws.dir) });
      const asked = await runFx(["ask", "--no-save", "say hi"], {
        cwd: ws.dir,
        env: { ...envFor(fake, ws.dir), FX_LOCAL_BASE_URL: "http://100.64.1.2:8080/v1" },
      });
      expect(asked.stderr + asked.stdout).toMatch(/Local/i);
      expect(asked.stdout).not.toContain("LOCAL_DIRECT_RESPONSE");
      expect(fake.chatRequests).toHaveLength(0);
    } finally {
      ws.cleanup();
      fake.stop();
    }
  });
});
