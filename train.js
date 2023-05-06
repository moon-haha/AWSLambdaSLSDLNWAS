const AWS = require("aws-sdk");
const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const s3 = new AWS.S3();
// const tensorboard = require("@tensorflow/tfjs-node").node.tensorBoard;
// const logdir = "/tmp/tensorboard_logs";
const zlib = require("zlib");

// .gz 파일 압축 해제 및 읽어오는 함수
function decompressAndReadFile(filePath) {
  return new Promise((resolve, reject) => {
    const buffer = fs.readFileSync(filePath);
    zlib.unzip(buffer, (err, decompressedBuffer) => {
      if (err) {
        reject(err);
      } else {
        resolve(decompressedBuffer);
      }
    });
  });
}
async function loadFashionMnistData() {
  const trainImages = await decompressAndReadFile(
    "./data/train-images-idx3-ubyte.gz"
  );
  const trainLabels = await decompressAndReadFile(
    "./data/train-labels-idx1-ubyte.gz"
  );
  const testImages = await decompressAndReadFile(
    "./data/t10k-images-idx3-ubyte.gz"
  );
  const testLabels = await decompressAndReadFile(
    "./data/t10k-labels-idx1-ubyte.gz"
  );

  return { trainImages, trainLabels, testImages, testLabels };
}

async function loadData() {
  const { trainImages, trainLabels } = await loadFashionMnistData();

  const imagesBuffer = new Uint8Array(trainImages.buffer, 16);
  const labelsBuffer = new Uint8Array(trainLabels.buffer, 8);

  const images = Array.from(imagesBuffer);
  const labels = Array.from(labelsBuffer);

  const imagesTensor = tf
    .tensor4d(images, [60000, 28, 28, 1], "float32")
    .div(tf.scalar(255));
  const labelsTensor = tf.oneHot(
    tf.tensor1d(labels.slice(0, 60000), "int32"),
    10
  );

  return [imagesTensor, labelsTensor];
}

async function loadTestData() {
  const { testImages, testLabels } = await loadFashionMnistData();

  const imagesBuffer = new Uint8Array(testImages.buffer, 16);
  const labelsBuffer = new Uint8Array(testLabels.buffer, 8);

  const images = Array.from(imagesBuffer);
  const labels = Array.from(labelsBuffer);

  const imagesTensor = tf
    .tensor4d(images, [10000, 28, 28, 1], "float32")
    .div(tf.scalar(255));
  const labelsTensor = tf.oneHot(
    tf.tensor1d(labels.slice(0, 10000), "int32"),
    10
  );

  return [imagesTensor, labelsTensor];
}

// async function uploadLogsToS3() {
//   const logFiles = fs.readdirSync(logDir).filter((file) => {
//     const filePath = `${logDir}/${file}`;
//     return fs.lstatSync(filePath).isFile(); // 디렉터리가 아닌 파일만 필터링
//   });
//   const bucketName = process.env.S3_BUCKET_NAME;

//   for (const file of logFiles) {
//     const fileBuffer = fs.readFileSync(`${logDir}/${file}`);

//     const params = {
//       Bucket: bucketName,
//       Key: `logs/${file}`,
//       Body: fileBuffer,
//     };

//     try {
//       const startTime = Date.now(); // 시작 시간 기록
//       await s3.upload(params).promise();
//       const elapsedTime = Date.now() - startTime; // 경과 시간 계산
//       console.log(
//         `Successfully uploaded ${file} to ${bucketName} in ${elapsedTime}ms`
//       );
//     } catch (error) {
//       console.error(`Error uploading ${file}: ${error}`);
//     }
//   }
// }

async function trainModel() {
  const model = tf.sequential();

  // Model architecture
  model.add(
    tf.layers.conv2d({
      inputShape: [28, 28, 1],
      filters: 16,
      kernelSize: 3,
      activation: "relu",
    })
  );
  model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

  model.add(
    tf.layers.conv2d({ kernelSize: 3, filters: 32, activation: "relu" })
  );

  model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
  model.add(
    tf.layers.conv2d({ kernelSize: 3, filters: 32, activation: "relu" })
  );
  model.add(tf.layers.flatten({}));

  model.add(tf.layers.dense({ units: 64, activation: "relu" }));
  model.add(tf.layers.dense({ units: 10, activation: "softmax" }));
  model.summary();

  // Model compile
  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  // Load and preprocess data
  const [trainImages, trainLabels] = await loadData();

  // Load test data
  const [testImages, testLabels] = await loadTestData();

  const batchSize = 128;
  const epochs = 30;
  // 시간 측정 시작
  const startTime = Date.now();

  // 모델 학습
  await model.fit(trainImages, trainLabels, {
    batchSize,
    epochs,
    callbacks: [
      {
        onEpochEnd: async (epoch, logs) => {
          console.log(
            `Epoch ${epoch + 1}: loss = ${logs.loss}, accuracy = ${logs.acc}`
          );
        },
      },
    ],
  });

  // 모델 평가
  const evaluation = model.evaluate(testImages, testLabels, { batchSize });
  console.log(
    `\n평가 결과:\n` +
      `  손실 = ${evaluation[0].dataSync()[0].toFixed(3)}; ` +
      `정확도 = ${evaluation[1].dataSync()[0].toFixed(3)}`
  );

  const loss = evaluation[0].dataSync()[0];
  const accuracy = evaluation[1].dataSync()[0];

  console.log(`Test Loss: ${loss}`);
  console.log(`Test Accuracy: ${accuracy}`);

  // 시간 측정 종료 및 출력
  const elapsedTime = Date.now() - startTime;
  console.log(`Training with Evaluation time: ${elapsedTime}ms`);

  // await uploadLogsToS3();

  // 모델 저장
  await model.save("file:///tmp/model");
  await uploadModelToS3();
}

async function uploadModelToS3() {
  const modelFiles = ["model.json", "weights.bin"];
  const bucketName = process.env.S3_BUCKET_NAME;

  for (const file of modelFiles) {
    const fileBuffer = fs.readFileSync(`/tmp/model/${file}`);

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
