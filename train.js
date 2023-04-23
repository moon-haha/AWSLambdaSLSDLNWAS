const tf = require("@tensorflow/tfjs-node");
const mnist = require("mnist");

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
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(
    tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: "relu" })
  );
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.flatten());
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
  const epochs = 5;

  // 모델 학습
  await model.fit(trainImages, trainLabels, {
    batchSize,
    epochs,
  });

  // 모델 저장
  await model.save("file://./model");
}

function loadData() {
  const dataset = mnist.set(60000, 0);
  const trainImages = dataset.training.map((item) => item.input);
  const trainLabels = dataset.training.map((item) => item.output);

  const xs = tf.tensor(trainImages).reshape([-1, 28, 28, 1]).div(255);
  const ys = tf.tensor(trainLabels);

  return [xs, ys];
}

trainModel();
