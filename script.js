/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

var body = document.querySelector("body");

const EXAMPLE_IMG = document.querySelector("#exampleImg");

const status = document.getElementById('status');
if (status) {
  status.innerText = 'Loaded TensorFlow.js - version: ' + tf.version.tfjs;
}

//tagName -> string, parent -> html document object, text -> string, 
//styles -> object exp: {position: "relative", left: "10px"}

const elementFactory = (tagName, parent, text, styles) => {
  let el = document.createElement(tagName);
  
  if(text){
    el.innerText = text;
  }
  
  if(styles){
    let styleString = "";
    
    for(let key in styles){
      styleString += key + ": " + styles[key] + "; ";
    }
    
    el.style = styleString;
  }
  
  parent.appendChild(el);
}

const MODEL_PATH = "https://tfhub.dev/google/tfjs-model/movenet/singlepose/lightning/4";

let movenet = undefined;

async function loadAndRunModel(){
  movenet = await tf.loadGraphModel(MODEL_PATH, { fromTFHub: true });
  
  let exampleInputTensor = tf.zeros([1, 192, 192, 3], "int32");
  let imageTensor = tf.browser.fromPixels(EXAMPLE_IMG);
  
  console.log(imageTensor.shape);
  
  let cropStartPoint = [15, 170, 0];
  let cropSize = [345, 345, 3];
  
  let croppedTensor = tf.slice(imageTensor, cropStartPoint, cropSize);
  
  let resizedTensor = tf.image.resizeBilinear(croppedTensor, [192, 192], true).toInt();
  
  console.log(resizedTensor.shape);
  
  let tensorOutput = movenet.predict(tf.expandDims(resizedTensor));
  let arratOutput = await tensorOutput.array();
  
  elementFactory("p", body, JSON.stringify(arratOutput), {position: "relative", top:"50px"});
}

loadAndRunModel();
