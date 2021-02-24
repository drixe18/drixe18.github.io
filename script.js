// const canvas = document.getElementById('mycanvas');
// let isPredicting = false;
const canvas = document.getElementById('mycanvas');
const img_sz = 256;

async function loadModel() {
  console.log('Loading SlimNet model..');
  const model = await tf.loadLayersModel('slimnet/model.json');
  // const model = await tf.loadGraphModel('sinet_fix/model.json'); // 90 ms
  // myModel.summary()
  console.log('Successfully loaded model');

  // warmup model!
  const img =  await getImage();
  const batch = await preprocessing(img);
  // console.log(batch);
  const resize = batch.resizeBilinear([img_sz, img_sz]);
  const expdim = resize.expandDims(0); 
  
  // Predict the model output
  const out = await model.predict(expdim);

  tf.dispose(out);
  img.dispose();
  batch.dispose();
  resize.dispose();
  expdim.dispose();
  return model;
 }

async function init() {
  try {
   webcam = await tf.data.webcam(document.getElementById('webcam'));
  } catch (e) {
   console.log(e);
   alert("No webcam found");
  }
 
  model = await loadModel();
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

// Perform mask feathering (Gaussian-blurring + Egde-smoothing)
function refine(mask) {

  const refine_out = tf.tidy(() => {
    // Reshape input
    const newmask = mask.reshape([1, img_sz, img_sz, 1]);

    //Gaussian kernel of size (7,7)
    const kernel = tf.tensor4d(
      [0.00092991, 0.00223073, 0.00416755, 0.00606375, 0.00687113, 0.00606375,
        0.00416755, 0.00223073, 0.00535124, 0.00999743, 0.01454618, 0.01648298,
        0.01454618, 0.00999743, 0.00416755, 0.00999743, 0.01867766, 0.02717584,
        0.03079426, 0.02717584, 0.01867766, 0.00606375, 0.01454618, 0.02717584,
        0.03954061, 0.04480539, 0.03954061, 0.02717584, 0.00687113, 0.01648298,
        0.03079426, 0.04480539, 0.05077116, 0.04480539, 0.03079426, 0.00606375,
        0.01454618, 0.02717584, 0.03954061, 0.04480539, 0.03954061, 0.02717584,
        0.00416755, 0.00999743, 0.01867766, 0.02717584, 0.03079426, 0.02717584,
        0.01867766], [7, 7, 1, 1]);

    // Convolve the mask with kernel   
    const blurred = tf.conv2d(newmask, kernel, strides = [1, 1], padding = 'same');
    //Reshape the output
    const fb = blurred.squeeze(0) 
    //Normalize the mask  to 0..1 range
    const norm_msk =   fb.sub(fb.min()).div(fb.max().sub(fb.min()))

    // Return the result
    return smoothstep(norm_msk);

});

return refine_out;
}

/* Smooth the mask edges */
function smoothstep(x) {

  const smooth_out = tf.tidy(() => {
  
    // Define the left and right edges 
    const edge0 = tf.scalar(0.3);
    const edge1 = tf.scalar(0.5);

    // Scale, bias and saturate x to 0..1 range
    const z = tf.clipByValue(x.sub(edge0).div(edge1.sub(edge0)), 0.0, 1.0);
    
    //Evaluate polynomial  z * z * (3 - 2 * x)
    return tf.square(z).mul(tf.scalar(3).sub(z.mul(tf.scalar(2))));
  
  });
  
   
  return smooth_out ;
}

function process(image, mask) {
  
  const blend_out = tf.tidy(() => {
  
   const img = image.resizeBilinear([300, 300]);
   const msk = refine(mask).resizeBilinear([300, 300]);;
   const img_crop = img.mul(msk);
  
   return img_crop;
  });
  return tf.cast(blend_out,'float32');
  
}

async function predict() {
  while (isPredicting) {
    var init_time = performance.now();
    // Capture the frame from the webcam.
    const img = await getImage();    
    const resize = img.resizeBilinear([img_sz, img_sz]);
    const img_clone = resize.toFloat()
    const batch = await preprocessing(resize);
    const expdim = batch.expandDims(0)
    // Predict the model output
    const out = await model.predict(expdim);
    // out.max().print();

    // Threshold the output to obtain mask
    const thresh = tf.scalar(0.9);
    const msk = tf.squeeze(out).greater(thresh);
    const cst = msk.toFloat().expandDims(-1);

    const blend = tf.tidy(() => {
      const mixed = tf.mul(img_clone, cst).div(tf.scalar(255.0));
      // const temp_sz = img_sz / 4
      // const temp_img = img_clone.resizeBilinear([temp_sz, temp_sz]);
      // const bkg_norm = temp_img.resizeBilinear([img_sz, img_sz]).div(tf.scalar(255.0));
      const bkg_norm = img_clone.div(tf.scalar(510.0));

      //Reverse mask: abs(pred_mask - 1)
      const rev_pred_mask = tf.abs(cst.sub(tf.scalar(1.)) );
      const bkg_matted = tf.mul( bkg_norm, rev_pred_mask );
      const res = tf.add(mixed, bkg_matted);
      return res;
     });
    
    // const smoothmask = smoothstep(cst);

    // Post-process the output and blend images
    // const blend = process(img_clone, cst).div(tf.scalar(255.0));
    // const blend = tf.mul(img_clone, cst).div(tf.scalar(255.0));

    await tf.browser.toPixels(blend, canvas);
  
    // Dispose all tensors 
    img.dispose();
    resize.dispose();
    img_clone.dispose();
    batch.dispose();
    expdim.dispose();
    out.dispose();
    msk.dispose();
    cst.dispose();
    blend.dispose();
    thresh.dispose();
    
    var elapsed_time = (performance.now() - init_time); //elapsed time in sec
    console.log(elapsed_time);
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
