const AWS = require("aws-sdk");
const fs = require("fs/promises");
const s3 = new AWS.S3();
async function downloadModelFromS3() {
  const modelFiles = ["model.json", "weights.bin"];
  const bucketName = process.env.S3_BUCKET_NAME;

  for (const file of modelFiles) {
    const params = {
      Bucket: bucketName,
      Key: `model/${file}`,
    };

    try {
      const startTime = Date.now(); // 시작 시간 기록
      const data = await s3.getObject(params).promise();
      await fs.writeFile(`./model/${file}`, data.Body);
      const elapsedTime = Date.now() - startTime; // 경과 시간 계산
      console.log(
        `Successfully downloaded ${file} from ${bucketName} in ${elapsedTime}ms`
      );
    } catch (error) {
      console.error(`Error downloading ${file}: ${error}`);
    }
  }
}

const tf = require("@tensorflow/tfjs-node");

async function loadModel() {
  await downloadModelFromS3();

  const model = await tf.loadLayersModel("file://./model/model.json");
  return model;
}

module.exports = { loadModel };
