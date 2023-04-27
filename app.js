require("dotenv").config();
const express = require("express");
const serverless = require("serverless-http");
const app = express();
const { loadModel } = require("./model");
const multer = require("multer");
const upload = multer();
const tf = require("@tensorflow/tfjs-node");
const bodyParser = require("body-parser");

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

app.post("/predict", upload.single("image"), async (req, res) => {
  //여기서 시간 측정 필요요함.
  if (!model) {
    res.status(500).send("Model not loaded");
    return;
  }

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

  res.json({ prediction: predictedClass });
});
const PORT = 3000;

app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});

module.exports.handler = serverless(app, {
  binary: ["*/*", "image/*"],
});
