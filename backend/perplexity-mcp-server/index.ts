#!/usr/bin/env node

import express, { type Request, type Response } from "express";
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse.js";
import {
  type CallToolResult,
  type ListToolsResult,
  McpError,
  ErrorCode,
  type Tool,
  ListToolsRequestSchema,
  CallToolRequestSchema,
  JSONRPCMessageSchema, // for debugging
} from "@modelcontextprotocol/sdk/types.js";

import dotenv from "dotenv";

dotenv.config();

/**
 * Definition of the Perplexity Ask Tool.
 * This tool accepts an array of messages and returns a chat completion response
 * from the Perplexity API, with citations appended to the message if provided.
 */
const PERPLEXITY_ASK_TOOL: Tool = {
  name: "perplexity_ask",
  description: `
  Ask Tool: Fast Latency - Quick and Accurate Fact-Based Responses

This tool provides rapid responses grounded firmly in verifiable truth, leveraging fast internet queries optimized for low latency and cost efficiency. It is ideal for straightforward Q&A tasks that require quick fact retrieval, simple calculations, brief historical facts, or concise explanations of widely understood topics. Use this tool for immediate, factual information needs where speed is prioritized over in-depth analysis or comprehensive reasoning.

**When to Use:**
- Simple math calculations, dates, and numerical data retrieval
- Basic historical fact-checking (e.g., dates, events, places)
- Straightforward definitions or explanations of well-established concepts

**When to Avoid:**
- Complex or nuanced topics requiring reasoning or multi-step analysis
- Questions requiring detailed citations or extensive research support
  `,
  inputSchema: {
    type: "object",
    properties: {
      messages: {
        type: "array",
        items: {
          type: "object",
          properties: {
            role: {
              type: "string",
              description:
                "Role of the message (e.g., system, user, assistant)",
            },
            content: {
              type: "string",
              description: "The content of the message",
            },
          },
          required: ["role", "content"],
        },
        description: "Array of conversation messages",
      },
    },
    required: ["messages"],
  },
};

/**
 * Definition of the Perplexity Reason Tool.
 * This tool performs reasoning queries using the Perplexity API.
 */
const PERPLEXITY_REASON_TOOL: Tool = {
  name: "perplexity_reason",
  description: `
Reasoning Tool: Medium Latency - Advanced Reasoning and Detailed Explanations

This high-performance tool employs sophisticated multi-step chain-of-thought (CoT) reasoning combined with advanced internet information retrieval to produce thorough, reasoned, and contextually rich answers. It is best suited for moderately complex tasks, topics requiring logical deductions, multi-step calculations, or detailed reasoning beyond simple facts. It balances speed with analytical depth, making it ideal for situations demanding thoughtful exploration of a question without the need for exhaustive citations.

**When to Use:**
- Moderately complex inquiries involving logical reasoning or explanations
- Multi-step problems such as algebraic calculations, scientific reasoning, or cause-and-effect analysis
- Tasks requiring synthesized understanding from multiple pieces of information

**When to Avoid:**
- Extremely simple, factual queries where \`perplexity_ask\` tool would suffice
- Very in-depth or scholarly research needing extensive citations and exhaustive detail
  `,
  inputSchema: {
    type: "object",
    properties: {
      messages: {
        type: "array",
        items: {
          type: "object",
          properties: {
            role: {
              type: "string",
              description:
                "Role of the message (e.g., system, user, assistant)",
            },
            content: {
              type: "string",
              description: "The content of the message",
            },
          },
          required: ["role", "content"],
        },
        description: "Array of conversation messages",
      },
    },
    required: ["messages"],
  },
};

/**
 * Definition of the Perplexity Research Tool.
 * This tool performs deep research queries using the Perplexity API.
 */
const PERPLEXITY_RESEARCH_TOOL: Tool = {
  name: "perplexity_research",
  description: `
  DeepResearch Tool: Very High Latency - Comprehensive Deep Research with Citations

This tool provides exhaustive, deeply researched answers supported by detailed citations, carefully curated from extensive internet searches and authoritative sources. Given its thorough nature, response times can extend up to 30 minutes. It is exclusively intended for queries explicitly requiring in-depth exploration, thorough source validation, and comprehensive reporting. Usage is mandatory whenever the user request explicitly contains the keywords "DeepResearch," "deepresearch," or "Deep Research."

**When to Use:**
- Extensive research tasks needing thorough analysis, critical evaluation, and cited evidence
- Complex scholarly or professional inquiries demanding rigorously sourced and verifiable data
- Any query explicitly labeled with "DeepResearch," "deepresearch," or "Deep Research"

**When to Avoid:**
- Queries needing rapid response or that involve relatively straightforward information
- Moderately complex tasks manageable by \`perplexity_reasoning\` tool's advanced reasoning capabilities  
  `,
  inputSchema: {
    type: "object",
    properties: {
      messages: {
        type: "array",
        items: {
          type: "object",
          properties: {
            role: {
              type: "string",
              description:
                "Role of the message (e.g., system, user, assistant)",
            },
            content: {
              type: "string",
              description: "The content of the message",
            },
          },
          required: ["role", "content"],
        },
        description: "Array of conversation messages",
      },
    },
    required: ["messages"],
  },
};

// Retrieve the Perplexity API key from environment variables
const PERPLEXITY_API_KEY = process.env.PERPLEXITY_API_KEY;
if (!PERPLEXITY_API_KEY) {
  console.error("Error: PERPLEXITY_API_KEY environment variable is required");
  process.exit(1);
}

/**
 * Performs a chat completion by sending a request to the Perplexity API.
 * Appends citations to the returned message content if they exist.
 *
 * @param {Array<{ role: string; content: string }>} messages - An array of message objects.
 * @param {string} model - The model to use for the completion.
 * @returns {Promise<string>} The chat completion result with appended citations.
 * @throws Will throw an error if the API request fails.
 */
async function performChatCompletion(
  messages: Array<{ role: string; content: string }>,
  model: string = "sonar-pro"
): Promise<string> {
  const url = new URL("https://api.perplexity.ai/chat/completions");
  const body = {
    model: model,
    messages: messages,
  };

  console.error(`Calling Perplexity API (model: ${model})...`);
  let response;
  try {
    response = await fetch(url.toString(), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${PERPLEXITY_API_KEY}`,
      },
      body: JSON.stringify(body),
    });
  } catch (error) {
    console.error("Network error calling Perplexity API:", error);
    throw new Error(`Network error while calling Perplexity API: ${error}`);
  }

  if (!response.ok) {
    let errorText = "Unknown API error";
    try {
      errorText = await response.text();
    } catch (parseError) {
      console.error("Failed to parse error response from Perplexity API");
    }
    console.error(
      `Perplexity API error: ${response.status} ${response.statusText}`,
      errorText
    );
    throw new Error(
      `Perplexity API error: ${response.status} ${response.statusText}\n${errorText}`
    );
  }

  let data;
  try {
    data = await response.json();
  } catch (jsonError) {
    console.error(
      "Failed to parse JSON response from Perplexity API:",
      jsonError
    );
    throw new Error(
      `Failed to parse JSON response from Perplexity API: ${jsonError}`
    );
  }

  console.error(`Perplexity API call successful (model: ${model}).`);
  let messageContent =
    data.choices[0]?.message?.content ?? "[No content received]";

  if (
    data.citations &&
    Array.isArray(data.citations) &&
    data.citations.length > 0
  ) {
    messageContent += "\n\nCitations:\n";
    data.citations.forEach((citation: any, index: number) => {
      messageContent += `[${index + 1}] ${JSON.stringify(citation)}\n`;
    });
  }

  return messageContent;
}

const server = new Server(
  {
    name: "perplexity-mcp-server",
    version: "0.1.0",
  },
  {
    capabilities: {
      // tools: {},
      tools: { listChanged: false },
      // Add capabilities the Python client sends during initialize
      sampling: {}, // Assuming empty object is sufficient declaration
      roots: { listChanged: true }, // Match what Python client sends
    },
  }
);

server.setRequestHandler(
  ListToolsRequestSchema,
  async (): Promise<ListToolsResult> => {
    console.error("Handling tools/list request");
    return {
      tools: [PERPLEXITY_ASK_TOOL, PERPLEXITY_REASON_TOOL],
    };
  }
);

server.setRequestHandler(
  CallToolRequestSchema,
  async (request): Promise<CallToolResult> => {
    const { name, arguments: args } = request.params;
    console.error(`Handling tools/call request for tool: ${name}`);

    try {
      if (!args || typeof args !== "object" || args === null) {
        throw new McpError(
          ErrorCode.InvalidParams,
          `Arguments must be an object for tool ${name}`
        );
      }
      if (!("messages" in args) || !Array.isArray(args.messages)) {
        throw new McpError(
          ErrorCode.InvalidParams,
          `Tool ${name} requires a 'messages' array argument`
        );
      }
      const messages = args.messages as Array<{
        role: string;
        content: string;
      }>;

      let resultText: string;
      switch (name) {
        case PERPLEXITY_ASK_TOOL.name:
          console.error(`Calling ${name} implementation...`);
          resultText = await performChatCompletion(messages, "sonar-pro");
          break;
        case PERPLEXITY_REASON_TOOL.name:
          console.error(`Calling ${name} implementation...`);
          resultText = await performChatCompletion(
            messages,
            "sonar-reasoning-pro"
          );
          break;
        default:
          console.error(`Unknown tool requested: ${name}`);
          throw new McpError(ErrorCode.MethodNotFound, `Unknown tool: ${name}`);
      }

      console.error(`Tool ${name} executed successfully.`);
      return {
        content: [{ type: "text", text: resultText }],
        isError: false,
      };
    } catch (error) {
      console.error(`Error executing tool ${name}:`, error);
      if (error instanceof McpError) {
        throw error;
      }
      return {
        content: [
          {
            type: "text",
            text: `Tool execution failed: ${
              error instanceof Error ? error.message : String(error)
            }`,
          },
        ],
        isError: true,
      };
    }
  }
);

const app = express();
const port = parseInt(process.env.PORT || "8080", 10);

const transports: { [sessionId: string]: SSEServerTransport } = {};

app.get("/sse", async (req: Request, res: Response) => {
  console.error(`SSE connection requested from ${req.ip}`);
  try {
    const transport = new SSEServerTransport("/messages", res);
    const logPrefix = `[${transport.sessionId}]`;

    transports[transport.sessionId] = transport;
    console.error(
      `${logPrefix} SSE transport created with sessionId: ${transport.sessionId}`
    );

    transport.onerror = (error: Error) => {
      console.error(`${logPrefix} SSEServerTransport received error:`, error);
    };

    res.on("close", () => {
      console.error(`${logPrefix} SSE connection closed.`);
      delete transports[transport.sessionId];
      transport
        .close()
        .catch((err: Error) =>
          console.error(`${logPrefix} Error closing transport:`, err)
        );
    });

    await server.connect(transport);
    console.error(`${logPrefix} Server connected to transport.`);
  } catch (error) {
    const errorPrefix = `[${
      (error as any)?.transport?.sessionId ?? "unknown-session"
    }]`;
    console.error(`${errorPrefix} Error setting up SSE connection:`, error);
    if (!res.headersSent) {
      res.status(500).send("Failed to establish SSE connection");
    } else {
      res.end();
    }
  }
});

app.post(
  "/messages",
  express.json({ limit: "5mb" }),
  async (req: Request, res: Response) => {
    const sessionId = req.query.sessionId as string;
    if (!sessionId) {
      console.error("POST /messages request missing sessionId query parameter");
      return res.status(400).send("Missing sessionId query parameter");
    }
    const logPrefix = `[${sessionId}]`;
    const transport = transports[sessionId];

    console.error(`${logPrefix} POST request received for /messages`);
    console.error(
      `${logPrefix} --> Received body:`,
      JSON.stringify(req.body, null, 2)
    );

    if (transport) {
      try {
        console.log(
          `${logPrefix} DEBUG: Attempting transport.handlePostMessage...`
        );
        await transport.handlePostMessage(req, res);
        console.log(
          `${logPrefix} DEBUG: transport.handlePostMessage potentially completed (response might already be sent).`
        );
      } catch (error) {
        console.error(
          `${logPrefix} ERROR caught explicitly in POST handler:`,
          error
        );
        if (!res.headersSent) {
          res.status(500).send("Internal Server Error handling message");
        }
      }
    } else {
      console.error(`${logPrefix} No active transport found.`);
      if (!res.headersSent) {
        res
          .status(404)
          .send(`No active transport found for sessionId: ${sessionId}`);
      }
    }
  }
);

app.get("/health", (req: Request, res: Response) => {
  console.error("Health check requested");
  res.status(200).send("OK");
});

app.listen(port, "0.0.0.0", () => {
  console.error(`Perplexity MCP Server (HTTP+SSE) listening on port ${port}`);
  console.error(` -> SSE connections on GET /sse`);
  console.error(` -> Client messages on POST /messages?sessionId=<sessionId>`);
  console.error(` -> Health check on GET /health`);
});
