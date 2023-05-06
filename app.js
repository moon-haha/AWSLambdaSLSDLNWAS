require("dotenv").config();
const express = require("express");
const serverless = require("serverless-http");
const app = express();
const { loadModel } = require("./model");
const multer = require("multer");
const upload = multer();
const tf = require("@tensorflow/tfjs-node");

const bodyParser = require("body-parser");

const { trainModel } = require("./train");
const AWS = require("aws-sdk");

AWS.config.update({
  region: "ap-northeast-2", // 사용하는 리전으로 변경
  accessKeyId: process.env.AWS_ACCESS_KEY_ID,
  secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
});

const lambda = new AWS.Lambda({
  region: "ap-northeast-2", // 사용하는 리전으로 변경
  accessKeyId: process.env.AWS_ACCESS_KEY_ID,
  secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
});

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

let trainColdStart = true;

// /train 엔드포인트 추가
app.get("/train", async (req, res) => {
  // AWS Lambda 환경 확인
  const isLambda = !!process.env.AWS_EXECUTION_ENV;
  const lambdaMemory = process.env.AWS_LAMBDA_FUNCTION_MEMORY_SIZE;
  const lambdaVCPU = lambdaMemory
    ? Math.floor(lambdaMemory / 3000) * 2
    : undefined;

  if (isLambda && trainColdStart) {
    console.log("This is a cold start for training");
  } else if (isLambda && !trainColdStart) {
    console.log("This is a warm start for training");
  } else {
    console.log("This is a local environment for training");
  }

  try {
    const startTime = Date.now();
    await trainModel();
    const elapsedTime = Date.now() - startTime;
    // 콜드 스타트 여부 업데이트
    if (isLambda && trainColdStart) {
      trainColdStart = false;
    }

    res.status(200).send({
      message: "Training completed",
      elapsedTime: elapsedTime,
      trainColdStart: trainColdStart,
      environment: isLambda ? "AWS Lambda" : "Local",
      coldStartTime: trainColdStart ? elapsedTime : undefined,
      warmStartTime: !trainColdStart ? elapsedTime : undefined,
      lambdaMemory: lambdaMemory, // 메모리 할당량 추가
      lambdaVCPU: lambdaVCPU, // vCPU 수 추가
    });
  } catch (error) {
    console.error(error);
    res.status(500).send("Training failed");
  }
});

let model;

//Cold Start 체크
let isColdStart = true;
app.post("/predict", upload.single("image"), async (req, res) => {
  // 시간 측정 시작
  const startTime = Date.now();
  // AWS Lambda 환경 확인
  const isLambda = !!process.env.AWS_EXECUTION_ENV;
  const lambdaMemory = process.env.AWS_LAMBDA_FUNCTION_MEMORY_SIZE;
  const lambdaVCPU = lambdaMemory
    ? Math.floor(lambdaMemory / 3000) * 2
    : undefined;

  if (isLambda && isColdStart) {
    console.log("This is a cold start");
  } else if (isLambda && !isColdStart) {
    console.log("This is a warm start");
  } else {
    console.log("This is a local environment");
  }
  // 모델 불러오기 시간 측정 시작
  const modelLoadStartTime = Date.now();

  if (!model) {
    model = await loadModel();
  }

  // 모델 불러오기 시간 측정 종료
  const modelLoadElapsedTime = Date.now() - modelLoadStartTime;

  // 이미지 파일 확인
  if (!req.file) {
    res.status(400).send("No image file provided");
    return;
  }

  // 이미지 전처리
  const buffer = req.file.buffer;
  const imgTensor = tf.node.decodeImage(buffer, 1);
  const resizedImg = tf.image.resizeBilinear(imgTensor, [28, 28]);
  const input = resizedImg.expandDims(0).div(255);

  // 모델 추론
  const prediction = model.predict(input);
  const predictedClass = prediction.argMax(-1).dataSync()[0];

  // 시간 측정 종료 및 결과 반환
  const elapsedTime = Date.now() - startTime;
  res.json({
    prediction: predictedClass,
    modelLoadElapsedTime: modelLoadElapsedTime,
    elapsedTime: elapsedTime,
    isColdStart: isColdStart,
    environment: isLambda ? "AWS Lambda" : "Local",
    coldStartTime: isColdStart ? elapsedTime : undefined,
    warmStartTime: !isColdStart ? elapsedTime : undefined,
    lambdaMemory: lambdaMemory, // 메모리 할당량 추가
    lambdaVCPU: lambdaVCPU, // vCPU 수 추가
  });
  // 콜드 스타트를 웜 스타트로 변경
  if (isLambda && isColdStart) {
    isColdStart = false;
  }
});

const isLambda = !!process.env.AWS_EXECUTION_ENV;
const PORT = 3000;

if (!isLambda) {
  app.listen(PORT, () => {
    console.log(`Server listening on port ${PORT}`);
  });
}

module.exports.handler = serverless(app, {
  binary: ["*/*", "image/*"],
});
