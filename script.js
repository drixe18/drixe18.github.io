// const canvas = document.getElementById('mycanvas');
// let isPredicting = false;
const canvas = document.getElementById('mycanvas');


async function loadModel() {
  console.log('Loading SINet model..');
  // const myModel = await tf.loadLayersModel('sinet/model.json');
  const model = await tf.loadGraphModel('sinet/model.json');
  // myModel.summary()
  console.log('Successfully loaded model');

  // warmup model!
  const img =  await getImage();
  const batch = await preprocessing(img);
  // console.log(batch);
  const resize = batch.resizeBilinear([224, 224]);
  const expdim = resize.expandDims(0); 
  // Predict the model output
  const out = await model.predict(expdim);

  
  // // warmup model!
  // const imgEl = document.getElementById('img');
  // const batch =  tf.browser.fromPixels(imgEl).toFloat().div(tf.scalar(127.5)).sub(tf.scalar(1)).expandDims();
  // const result1 = await myModel.predict(batch);
  // console.log(out);
  tf.dispose(out);
  img.dispose();
  batch.dispose();
  resize.dispose();
  expdim.dispose();
  return model;
 }

async function checkModel() {
    console.log('Loading SINet model..');
    // const myModel = await tf.loadLayersModel('sinet/model.json');
    const myModel = await tf.loadGraphModel('sinet/model.json');

    // myModel.summary()
    console.log('Successfully loaded model');

    // Make a prediction through the model on our image.
    const imgEl = document.getElementById('img');
    const batch =  tf.browser.fromPixels(imgEl).toFloat().div(tf.scalar(127.5)).sub(tf.scalar(1)).expandDims();
    const result1 = await myModel.predict(batch);
    console.log(result1);
    const result2 = await myModel.predict(batch);
    console.log(result2);
    const result3 = await myModel.predict(batch);
    console.log(result3);

    await tf.browser.toPixels(tf.squeeze(result1), canvas);
    tf.dispose(result1);
    tf.dispose(result2);
    tf.dispose(result3);
}

async function init() {
  try {
   webcam = await tf.data.webcam(document.getElementById('webcam'));
  } catch (e) {
   console.log(e);
   alert("No webcam found");
  }
 
  model = await loadModel();
 
  // bgim = loadBackground();
 
  const screenShot = await webcam.capture();
  // const pred = model.predict(tf.zeros([1, 128, 128, 3]).toFloat());
 
  // var readable_output = pred.dataSync();
  // console.log(readable_output);
  // console.log(model.summary());
 
  // pred.dispose();
  screenShot.dispose();
  console.log('Initted!')
 
}

async function getImage() {
  const frame = await webcam.capture();
  return frame;
}

async function preprocessing(img) {
  const processedImg = 
    tf.tidy(() => img.div(tf.scalar(127.5)).sub(tf.scalar(1)));
  img.dispose();
  return processedImg;
}

function process(image, mask) {
  
  const blend_out = tf.tidy(() => {
  
   const img = image.resizeBilinear([300, 300]);
   const msk = refine(mask).resizeNearestNeighbor([300, 300]);;
   const img_crop = img.mul(msk);
  //  const bgd_crop = bgim.mul(tf.scalar(1.0).sub(msk));
  //  const result = tf.add(img_crop, bgd_crop);
  
   return img_crop;
  });
  return tf.cast(blend_out,'float32');
  
}

async function predict() {
  while (isPredicting) {
    // Capture the frame from the webcam.
    const frame =  await getImage();
    // frame.print();
    
    const img_clone = frame.resizeBilinear([224, 224]);
    const img = await preprocessing(frame);
    
    const resize = img.resizeBilinear([224, 224]);
    const expdim = resize.expandDims(0);
 
    // Predict the model output
    const out = await model.predict(expdim);
    // out.max().print();

    // Threshold the output to obtain mask
    const thresh = tf.scalar(0.5);
    // const msk = tf.squeeze(out).greater(thresh);
    const msk = tf.squeeze(out);
    const cst = msk.toFloat().expandDims(-1);

    // Post-process the output and blend images
    // const blend = process(img_clone, cst).div(tf.scalar(255.0));
    const blend = tf.mul(img_clone, cst).div(tf.scalar(255.0));
    // Draw output on the canvas
    // let canvas = document.getElementById('mycanvas');

    await tf.browser.toPixels(blend, canvas);
  
    // Dispose all tensors 
    img_clone.dispose();
    frame.dispose();
    blend.dispose();
    resize.dispose();
    msk.dispose();
    expdim.dispose();
    cst.dispose();
    thresh.dispose();
    out.dispose();
    img.dispose();
 
    // Wait for next frame
    await tf.nextFrame();
  
  }
 
}

var el = document.getElementById('start');
if(el){
  el.addEventListener('click', async () => {
    isPredicting = true;
    predict();
   });
}

/* Set up the on-stop listener */
document.getElementById('stop').addEventListener('click', () => {
  isPredicting = false;
 });

init();
