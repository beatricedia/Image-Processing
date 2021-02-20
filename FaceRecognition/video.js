let video = document.getElementById("videoInput"); // video is the id of video tag
// navigator.mediaDevices.getUserMedia({ video: true, audio: false })
//     .then(function(stream) {
//         video.srcObject = stream;
//         video.play();
//     })
//     .catch(function(err) {
//         console.log("An error occurred! " + err);
//     });
    // let src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
let dst = new cv.Mat(video.height, video.width, cv.CV_8UC1);
// let canvasFrame = document.getElementById("canvasFrame"); // canvasFrame is the id of <canvas>
// let context = canvasFrame.getContext("2d");
// let cap = new cv.VideoCapture(video);

// const FPS = 30;
// function processVideo() {
//     try {
//         if (!streaming) {
//             // clean and stop.
//             src.delete();
//             dst.delete();
//             return;
//         }
//         let begin = Date.now();
//         // start processing.
//         cap.read(src);
//         cv.cvtColor(src, dst, cv.COLOR_RGBA2GRAY);
//         cv.imshow('canvasOutput', dst);
//         // schedule the next one.
//         let delay = 1000/FPS - (Date.now() - begin);
//         setTimeout(processVideo, delay);
//     } catch (err) {
//         utils.printError(err);
//     }
// };

// // schedule the first one.
// setTimeout(processVideo, 0);



// let utils = new Utils('errorMessage');

// utils.loadCode('codeSnippet', 'codeEditor');

// let streaming = false;
// let videoInput = document.getElementById('videoInput');
// let startAndStop = document.getElementById('startAndStop');
// let canvasOutput = document.getElementById('canvasOutput');
// let canvasContext = canvasOutput.getContext('2d');

// startAndStop.addEventListener('click', () => {
//     if (!streaming) {
//         utils.clearError();
//         utils.startCamera('qvga', onVideoStarted, 'videoInput');
//     } else {
//         utils.stopCamera();
//         onVideoStopped();
//     }
// });

// function onVideoStarted() {
//     streaming = true;
//     startAndStop.innerText = 'Stop';
//     videoInput.width = videoInput.videoWidth;
//     videoInput.height = videoInput.videoHeight;
//     utils.executeCode('codeEditor');
// }

// function onVideoStopped() {
//     streaming = false;
//     canvasContext.clearRect(0, 0, canvasOutput.width, canvasOutput.height);
//     startAndStop.innerText = 'Start';
// }

// utils.loadOpenCv(() => {
//     startAndStop.removeAttribute('disabled');
// });

