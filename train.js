const AWS = require("aws-sdk");
const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const mnist = require("mnist");
const s3 = new AWS.S3();
const tensorboard = require("@tensorflow/tfjs-node").node.tensorBoard;
const logDir = "./logs";

async function uploadLogsToS3() {
  const logFiles = fs.readdirSync(logDir).filter((file) => {
    const filePath = `${logDir}/${file}`;
    return fs.lstatSync(filePath).isFile(); // 디렉터리가 아닌 파일만 필터링
  });
  const bucketName = process.env.S3_BUCKET_NAME;

  for (const file of logFiles) {
    const fileBuffer = fs.readFileSync(`${logDir}/${file}`);

    const params = {
      Bucket: bucketName,
      Key: `logs/${file}`,
      Body: fileBuffer,
    };

    try {
      const startTime = Date.now(); // 시작 시간 기록
      await s3.upload(params).promise();
      const elapsedTime = Date.now() - startTime; // 경과 시간 계산
      console.log(
        `Successfully uploaded ${file} to ${bucketName} in ${elapsedTime}ms`
      );
    } catch (error) {
      console.error(`Error uploading ${file}: ${error}`);
    }
  }
}

async function trainModel() {
  const model = tf.sequential();

  // 모델 아키텍처
  model.add(
    tf.layers.conv2d({
      inputShape: [28, 28, 1],
      filters: 16,
      kernelSize: 3,
      activation: "relu",
    })
  );
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

  model.add(
    tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: "relu" })
  );
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 64, activation: "relu" }));
  model.add(tf.layers.dropout({ rate: 0.5 }));
  model.add(tf.layers.dense({ units: 10, activation: "softmax" }));

  const optimizer = tf.train.adam(0.001);

  // 모델 컴파일
  model.compile({
    optimizer: optimizer,
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  // 데이터 로드 및 전처리
  const [trainImages, trainLabels] = loadData();

  // 테스트 데이터 로드
  const [testImages, testLabels] = loadTestData();

  const batchSize = 128;
  const epochs = 5;

  // 시간 측정 시작
  const startTime = Date.now();

  // 모델 학습
  await model.fit(trainImages, trainLabels, {
    batchSize,
    epochs,
    callbacks: [
      // {
      //   onEpochEnd: async (epoch, logs) => {
      //     console.log(
      //       `Epoch ${epoch + 1}: loss = ${logs.loss}, accuracy = ${logs.acc}`
      //     );
      //   },
      // },
      tensorboard(logDir),
    ],
  });
  // 모델 평가
  const evaluation = model.evaluate(testImages, testLabels, { batchSize });
  const loss = evaluation[0].dataSync()[0];
  const accuracy = evaluation[1].dataSync()[0];

  console.log(`Test Loss: ${loss}`);
  console.log(`Test Accuracy: ${accuracy}`);

  // 시간 측정 종료 및 출력
  const elapsedTime = Date.now() - startTime;
  console.log(`Training time: ${elapsedTime}ms`);

  await uploadLogsToS3();

  // 모델 저장
  await model.save("file://./model");
  await uploadModelToS3();
}

function loadData() {
  const dataset = mnist.set(56000, 14000);
  const trainImages = dataset.training.map((item) => item.input);
  const trainLabels = dataset.training.map((item) => item.output);

  const xs = tf.tensor(trainImages).reshape([-1, 28, 28, 1]).div(255);
  const ys = tf.tensor(trainLabels);

  return [xs, ys];
}

function loadTestData() {
  const dataset = mnist.set(56000, 14000);
  const testImages = dataset.test.map((item) => item.input);
  const testLabels = dataset.test.map((item) => item.output);

  const xs = tf.tensor(testImages).reshape([-1, 28, 28, 1]).div(255);
  const ys = tf.tensor(testLabels);

  return [xs, ys];
}

async function uploadModelToS3() {
  const modelFiles = ["model.json", "weights.bin"];
  const bucketName = process.env.S3_BUCKET_NAME;

  for (const file of modelFiles) {
    const fileBuffer = fs.readFileSync(`./model/${file}`);

    const params = {
      Bucket: bucketName,
      Key: `model/${file}`,
      Body: fileBuffer,
    };

    try {
      const startTime = Date.now(); // 시작 시간 기록
      await s3.upload(params).promise();
      const elapsedTime = Date.now() - startTime; // 경과 시간 계산
      console.log(
        `Successfully uploaded ${file} to ${bucketName} in ${elapsedTime}ms`
      );
    } catch (error) {
      console.error(`Error uploading ${file}: ${error}`);
    }
  }
}

module.exports = { trainModel };
