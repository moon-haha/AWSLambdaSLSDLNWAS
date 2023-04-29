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

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

/**
 * routes
 */
app.get("/", (req, res) => {
  res.send("Hello, World!");
});

let model;
loadModel().then((loadedModel) => {
  model = loadedModel;
});

let trainColdStart = true;

// /train 엔드포인트 추가
app.get("/train", async (req, res) => {
  // AWS Lambda 환경 확인
  const isLambda = !!process.env.AWS_EXECUTION_ENV;

  if (isLambda && trainColdStart) {
    console.log("This is a cold start for training");
  }

  try {
    const startTime = Date.now();
    await trainModel();
    const elapsedTime = Date.now() - startTime;
    // 콜드 스타트 여부 업데이트
    if (isLambda && trainColdStart) {
      trainColdStart = false;
    }

    res
      .status(200)
      .send({
        message: "Training completed",
        elapsedTime: elapsedTime,
        trainColdStart: trainColdStart,
      });
  } catch (error) {
    console.error(error);
    res.status(500).send("Training failed");
  }
});

//Cold Start 체크
let isColdStart = true;
app.post("/predict", upload.single("image"), async (req, res) => {
  // 시간 측정 시작
  const startTime = Date.now();

  // AWS Lambda 환경 확인
  const isLambda = !!process.env.AWS_EXECUTION_ENV;

  if (isLambda && isColdStart) {
    console.log("This is a cold start");
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
  });
});
const PORT = 3000;

app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});

module.exports.handler = serverless(app, {
  binary: ["*/*", "image/*"],
});
