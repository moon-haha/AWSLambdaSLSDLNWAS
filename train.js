const AWS = require("aws-sdk");
const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const mnist = require("mnist");
const s3 = new AWS.S3();
async function trainModel() {
  const model = tf.sequential();

  // 모델 아키텍처
  model.add(
    tf.layers.conv2d({
      inputShape: [28, 28, 1],
      filters: 32,
      kernelSize: 3,
      activation: "relu",
    })
  );
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

  model.add(
    tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: "relu" })
  );
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

  model.add(
    tf.layers.conv2d({ filters: 128, kernelSize: 3, activation: "relu" })
  );
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 256, activation: "relu" }));
  model.add(tf.layers.dropout({ rate: 0.5 }));
  model.add(tf.layers.dense({ units: 128, activation: "relu" }));
  model.add(tf.layers.dropout({ rate: 0.5 }));
  model.add(tf.layers.dense({ units: 10, activation: "softmax" }));

  // 모델 컴파일
  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  // 데이터 로드 및 전처리
  const [trainImages, trainLabels] = loadData();

  const batchSize = 128;
  const epochs = 30;

  // 시간 측정 시작
  const startTime = Date.now();

  // 모델 학습
  await model.fit(trainImages, trainLabels, {
    batchSize,
    epochs,
  });

  // 시간 측정 종료 및 출력
  const elapsedTime = Date.now() - startTime;
  console.log(`Training time: ${elapsedTime}ms`);

  // 모델 저장
  await model.save("file://./model");
  await uploadModelToS3();
}

function loadData() {
  const dataset = mnist.set(60000, 0);
  const trainImages = dataset.training.map((item) => item.input);
  const trainLabels = dataset.training.map((item) => item.output);

  const xs = tf.tensor(trainImages).reshape([-1, 28, 28, 1]).div(255);
  const ys = tf.tensor(trainLabels);

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
