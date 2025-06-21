import { Hono } from "hono";
import { YoloDetectionInference } from "ppu-yolo-onnx-inference";
import { COCO128_CLASS } from "./coco128-class";
import index from "./index.html";

const modelFile = Bun.file("./coco128.onnx");
const modelBuffer = await modelFile.arrayBuffer();

const model = new YoloDetectionInference({
  model: {
    onnx: modelBuffer,
    classNames: COCO128_CLASS,
  },
  thresholds: {
    confidence: 0.5,
  },
});

console.log("🤖 Initializing YOLO model...");
await model.init();
console.log("✅ YOLO model initialized successfully");

const app = new Hono();

app.use("/detect", async (c, next) => {
  c.header("Access-Control-Allow-Origin", "*");
  c.header("Access-Control-Allow-Methods", "POST, OPTIONS");
  c.header("Access-Control-Allow-Headers", "Content-Type");

  if (c.req.method === "OPTIONS") {
    return c.text("", 200);
  }

  await next();
});

app.post("/detect", async (c) => {
  try {
    const body = await c.req.parseBody();
    const imageFile = body["image"] as File;
    const confidence = parseFloat(body["confidence"] as string) || 0.5;

    if (!imageFile) {
      return c.json(
        {
          success: false,
          error: "No image provided",
          detections: [],
        },
        400
      );
    }

    const arrayBuffer = await imageFile.arrayBuffer();
    const startTime = Date.now();

    const detections = await model.detect(arrayBuffer);
    const inferenceTime = Date.now() - startTime;

    console.log(
      `🔍 Detected ${detections.length} objects in ${inferenceTime}ms`
    );

    if (detections.length > 0) {
      detections.forEach((detection, index) => {
        console.log(
          `  ${index + 1}. ${detection.className} (${(
            detection.confidence * 100
          ).toFixed(1)}%) at [${detection.box.x}, ${detection.box.y}, ${
            detection.box.width
          }, ${detection.box.height}]`
        );
      });
    }

    return c.json({
      success: true,
      detections: detections.filter((d) => d.confidence >= confidence),
      inferenceTime: inferenceTime,
    });
  } catch (error) {
    console.error("❌ Detection error:", error);

    return c.json(
      {
        success: false,
        error:
          error instanceof Error ? error.message : "Unknown error occurred",
        detections: [],
      },
      500
    );
  }
});

// Graceful shutdown
process.on("SIGINT", async () => {
  console.log("\\n🛑 Shutting down server...");
  await model.destroy();
  console.log("✅ Model destroyed successfully");
  process.exit(0);
});

console.log("🎥 Live Object Detection Server running at http://localhost:3000");
console.log("🤖 YOLO model ready for real-time inference");
console.log("📱 Make sure to allow camera permissions when prompted");

export default {
  port: 3000,
  routes: {
    "/": index,
  },
  fetch: app.fetch,
};
