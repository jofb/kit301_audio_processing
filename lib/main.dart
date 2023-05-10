import 'dart:ffi';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:scidart/numdart.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'dart:math';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:io';
import 'package:ffmpeg_kit_flutter_audio/ffmpeg_kit.dart';
import 'package:ffmpeg_kit_flutter_audio/ffprobe_kit.dart';
import 'package:fftea/fftea.dart';
import 'package:fftea/stft.dart';
import 'audio_processing.dart';
import 'package:wav/wav.dart';
import 'package:ml_linalg/linalg.dart';
// import 'package:dart_tensor/dart_tensor.dart';
import 'package:scidart/scidart.dart';

void main() {
  runApp(const MainApp());
}

class MainApp extends StatelessWidget {
  const MainApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'LID Inference',
      home: Scaffold(
        body: Center(
          child: languageIdWidget(),
        ),
      ),
    );
  }
}

class languageIdWidget extends StatefulWidget {
  const languageIdWidget({
    super.key,
  });

  @override
  State<languageIdWidget> createState() => _languageIdWidgetState();
}

class _languageIdWidgetState extends State<languageIdWidget> {
  String identifiedLanguage = "";

  // mel spectrum constants.
  final MEL_BREAK_FREQUENCY_HERTZ = 700.0;
  final MEL_HIGH_FREQUENCY_Q = 1127.0;

  int msFrames(int sampleRate, int ms) {
    return ((sampleRate.toDouble()) * 1e-3 * (ms.toDouble())).toInt();
  }

  List<double> hertzToMel(List<double> frequencies) {
    // this is meant to be list mapping
    return frequencies
        .map((frequency) =>
            MEL_HIGH_FREQUENCY_Q *
            log(1.0 + (frequency / MEL_BREAK_FREQUENCY_HERTZ)))
        .toList();
  }

  Matrix calculateMelWeightsMatrix(
      {int numMelBins = 20,
      numSpectrogramBins = 129,
      sampleRate = 8000,
      lowerEdgeHertz = 125.0,
      upperEdgeHertz = 3800.0}) {
    // TODO PARAMETER VALIDATION (IGNORE FOR NOW)
    //   _validate_arguments(num_mel_bins, sample_rate,
    //                       lower_edge_hertz, upper_edge_hertz, dtype)

    // cast samplerate to float32
    // cast loweredgehertz to float32 tensor
    // cast upperedgehertz to float32 tensor
    // create zeros tensor of float32

    const bandsToZero = 1;
    List<double> linearFrequencies = [];

    // create linspace from 0 to nyquistHertz
    var nyquistHertz = sampleRate / 2.0;
    var stepValue = nyquistHertz / numSpectrogramBins;
    for (var stepCount = 1; stepCount <= numSpectrogramBins; stepCount++) {
      linearFrequencies.add(stepValue * stepCount);
    }

    // slice from bandsToZero, we will add this back later
    linearFrequencies = linearFrequencies.sublist(bandsToZero);

    // convert to mel scale from hertz
    List<double> spectrogramBinMel = hertzToMel(linearFrequencies);

    // create the linspace for band edges
    List<double> bandEdgesMelLinSpace = [];
    // start at htm(lower), go to htm(upper), step by num_mel_bens + 2,
    var lowerEdgeMelLin = hertzToMel([lowerEdgeHertz])[0];
    var upperEdgeMelLin = hertzToMel([upperEdgeHertz])[0];
    var edgeStep = (upperEdgeMelLin - lowerEdgeMelLin) / (numMelBins + 2);
    for (var i = 0; i < (numMelBins + 2); i++) {
      bandEdgesMelLinSpace.add(edgeStep * i);
    }

    // create the triples in one array
    List<List<double>> bandEdges = [];
    // now create a frame using (bandEdgesMelLinSpace, frame_length = 3, and frame_step = 1)
    for (var i = 1; i < (numMelBins + 1); i++) {
      // create a new triple with i-1 as lower, i as center, i+1 as upper
      List<double> newlist = [
        bandEdgesMelLinSpace[i - 1],
        bandEdgesMelLinSpace[i],
        bandEdgesMelLinSpace[i + 1]
      ];
      bandEdges.add(newlist);
    }

    // TODO this could be refactored, likely don't need seperate arrays for each one
    List<double> lowerMel = [];
    List<double> centerMel = [];
    List<double> upperMel = [];

    for (int i = 0; i < bandEdges.length; i++) {
      // continue up to num_mel_bins - 2
      lowerMel.add(bandEdges[i][0]);
      // continue up to num_mel_bins - 1
      centerMel.add(bandEdges[i][1]);
      // continue up to bandEdges.length
      upperMel.add(bandEdges[i][2]);
    }

    /******** Calculating the slopes *********/
    // lower slope
    // a = (centerMel - lowerMel)
    List<double> a = [];
    for (var i = 0; i < lowerMel.length; i++) {
      a.add(centerMel[i] - lowerMel[i]);
    }

    // (spectrogramBinMel - lowerMel) / a
    List<List<double>> lowerSlopes = [];
    for (var i = 0; i < spectrogramBinMel.length; i++) {
      lowerSlopes.add([]);
      for (var j = 0; j < lowerMel.length; j++) {
        // subtract each element
        lowerSlopes[i].add((spectrogramBinMel[i] - lowerMel[j]) / a[j]);
      }
    } // (256, 40) / (1, 40)

    // upper slope
    // b = (upperEdgeMel - centerMel)
    List<double> b = [];
    for (var i = 0; i < upperMel.length; i++) {
      b.add(upperMel[i] - centerMel[i]);
    }
    // (upperEdgeMel - spectrogramBinMel) / b
    List<List<double>> upperSlopes = [];
    for (var i = 0; i < spectrogramBinMel.length; i++) {
      upperSlopes.add([]);
      for (var j = 0; j < upperMel.length; j++) {
        // subtract each element
        upperSlopes[i].add((upperMel[j] - spectrogramBinMel[i]) / b[j]);
      }
    } // (256, 40) / (1, 40) = (256, 40)

    // create the final mel weights matrix
    List<List<double>> melWeightsMatrix = [];
    for (var i = 0; i < lowerSlopes.length; i++) {
      melWeightsMatrix.add([]);
      // 256
      for (var j = 0; j < lowerSlopes[0].length; j++) {
        // 40
        var minVal = min(upperSlopes[i][j], lowerSlopes[i][j]);
        melWeightsMatrix[i].add(minVal > 0 ? minVal : 0);
      }
    }

    // Re-add the zeroed lower bins we sliced out above.
    melWeightsMatrix.insert(0, List.filled(melWeightsMatrix[0].length, 0.0));

    // we can either return this as a List<List<double>> or convert ot a Matrix here
    // for now lets return a matrix
    return Matrix.fromList(melWeightsMatrix);
  }

  Future<void> identifyAudio() async {
    /******************* SPECTROGRAM STUFF *******************/
    // the resampled audio is in the assets folder under filtered_audio.wav

    // 1. create normal spectrogram based on lidbox implementation
    // spectrograms(signal, rate)
    const sampleRate = 16000;
    const frameLengthMS = 25;
    const frameStepMS = 10;
    const power = 2.0;
    const fmin = 0.0;
    const fmax = 8000.0;
    const fftLength = 512;

    var frameLength = msFrames(sampleRate, frameLengthMS);
    var frameStep = msFrames(sampleRate, frameStepMS);

    List<double> audio = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    List<double> aud = List.filled(512, 0.00034);
    // (10000) / 160 = num_frames

    // this block reads in the resampled audio from assets, saves it to cache, then reads it as a wav file
    final asset = await rootBundle.load("assets/filtered_audio.wav");
    final tempDir = await getTemporaryDirectory();
    final file = File("${tempDir.path}/mysoundfile.wav");
    await file.writeAsBytes(asset.buffer.asUint8List());

    final oceanWaves = await Wav.readFile("${tempDir.path}/mysoundfile.wav");

    //final owaves2 = await getPCMData("${tempDir.path}/mysoundfile.wav");
    //print(owaves2.length);
    // aud = owaves2;
    aud = oceanWaves.toMono();

    // now save to cache

    // final buffer = asset.buffer;

    //final oceanWaves = await Wav.readFile('assets/filtered_audio.wav');
    //print(oceanWaves);
    // print(oceanWaves.format);

    /******************* POWER SPECTROGRAM *****/
    /*** zero-padding and you (this is unrelated to the problem down below (this one is solved))
 
    the windowing function takes frames of length 400, then steps 160 forward, and then takes another frame of 400 
    i.e first frame from 0-400, second frame from 160-560, etc.
    
    if there are length mismatches 
    i.e we have an array of length 530, 
    so it starts at 400, tries to step 160 past that but can't since its out of bounds (400 + 160 = 560 > 530)
    
    then there are varying behaviours to deal with this issue
    
    the python stft simply cuts out anything that doesn't fit cleanly in its steps
    so for the 530 example it would cut out everything past 400, and just run the stft with one frame
    
    the dart stft pads anything that doesn't fit cleanly with zeros
    for the 530 example it would pad our audio with zeros until its length 560, and run the stft with two frames
    
    we need to mimic the behaviour of the python impl. by clamping the audio at the end point so that it fits cleanly with steps
    
    TODO have not tested what happens with audio length < 400, but this should never occur
     */

    final int audioLength = aud.length;
    List<double> clampedAudio =
        aud.sublist(0, audioLength - ((audioLength - frameLength) % frameStep));

    /* PROBLEM:
      the chunkSize of the frames needs to be frameLength (400),
      however the python impl. is sneaky and applies ffts of size 512 on the audio
      what this means is that it tries to apply an fft on the frame (which will be length 400) with an fft function thats been told the length is 512
      as you may recall, the python impl. simply zero pads out the frame when doing an fft to ensure they are equal
      this is a similar problem to the one i had with the inverse fft
      so we may need to rewrite the stft.run, and ensure that each frame is zero padded out, while using a different fft that is of length 512
      (while still keeping chunks of 400....)
    
     */
    //final stft = STFT(fftLength, Window.hanning(fftLength));
    var spectrogram = <Float64List>[];

    // clampedAudio =
    //     clampedAudio.sublist(0, 23441); // > 23440 we start getting infinity
    // print(clampedAudio[23440]);

    // 23440 - 400 = 23040
    // 23040 / 160 = 144

    customSTFT(clampedAudio, (Float64x2List freq) {
      spectrogram.add(freq.discardConjugates().magnitudes());
    }, frameLength, frameStep, fftLength,
        Window.hanning(frameLength)); //Window.hanning(frameLength)

    // this is our output for a pow spectrograms
    // TODO fix this, get rid of powspec and rename powpow
    List<List<double>> powpow = [];
    spectrogram.forEach((element) {
      powpow.add(element.map((e) => pow(e.abs(), power).toDouble()).toList());
    });

    //print(powpow[0]);

    //print(powpow[0].length);

    /******************* MEL SPECTROGRAM MATRIX WEIGHTS *****/

    var melWeights = calculateMelWeightsMatrix(
        numMelBins: 40,
        numSpectrogramBins: powpow[0].length,
        sampleRate: sampleRate,
        lowerEdgeHertz: fmin,
        upperEdgeHertz: fmax);

    /******************LINEAR TO MEL SPECTROGRAM */

    // mel_weights = tf.signal.linear_to_mel_weight_matrix(
    //     num_mel_bins=num_mel_bins,
    //     num_spectrogram_bins=tf.shape(S)[2],
    //     sample_rate=sample_rate,
    //     lower_edge_hertz=fmin,
    //     upper_edge_hertz=fmax)
    // return tf.tensordot(S, mel_weights, 1)

    var specMatrix = Matrix.fromList(powpow);

    // NOT SURE IF THIS IS EQUIVALENT TO A DOT (IM STUPID MATRIX MATH)
    var spectrogramMatrix = specMatrix * melWeights;

    // spectrogram.toList().forEach((element) {
    //   element.forEach((e) {
    //     if (e.isInfinite) {
    //       print("hrewqhoriqw $element");
    //     }
    //   });
    // });

    //print("Spectrogram Matrix:  $spectrogramMatrix");

    /******************* LOG SCALE ********************/

    // var mynewmatrix = spectrogramMatrix;
    var mynewmatrix =
        spectrogramMatrix.mapElements((element) => log(element + 1e-06));

    // var countedInfinities = 0;
    // melWeights.forEach((element) {
    //   for (var i = 0; i < element.length; i++) {
    //     if (element.toList()[i].isInfinite) {
    //       countedInfinities++;
    //     }
    //   }
    // });
    // print("Infinite rows: $countedInfinities");

    // try only using one row of the matrix
    // var testmatrix = mynewmatrix.toList().sublist(21);
    // INFINITY ON 21ST ELEMENT

    // TWO PROBLEMS
    // 1. INPUTS WRONG
    // 2. MODEL DOESNT LIKE THE INPUT

    /******************* MODEL STUFF *******************/
    final interpreter = await Interpreter.fromAsset('model.tflite');
    // grab shapes
    // TODO figure out resizing
    // shape: (num_batch, num_frames, 40)
    interpreter.resizeInputTensor(0, [1, mynewmatrix.rows.length, 40]);
    // interpreter.resizeInputTensor(0, [1, testmatrix.length, 40]);
    interpreter.allocateTensors();
    var inputShape = interpreter.getInputTensor(0).shape;
    var outputShape = interpreter.getOutputTensor(0).shape;
    // set shapes on dummy data and output tensor
    // var input = List.filled(40, Random().nextDouble()).reshape(inputShape);
    // reshape spectrogrammatrix to fit shape (num_frames, 40) -> (1, num_frames, 40)
    // var input = mynewmatrix.toList().reshape(inputShape);
    var input = mynewmatrix.toList().reshape(inputShape);
    var output = List.filled(4, 0).reshape(outputShape);
    // run interpreter
    interpreter.run(input, output);
    print(output);

    var languages = ["Estonian", "Mongolian", "Tamil", "Turkish"];
    List<double> convertedOutput = output[0] as List<double>;

    var THEONELANG =
        languages[convertedOutput.indexOf(convertedOutput.reduce(max))];
    setState(() {
      identifiedLanguage = THEONELANG;
    });
  }

  void playAudio() async {
    final tempDir = await getTemporaryDirectory();

    final asset = await rootBundle.load("assets/filtered_audio.wav");
    final file = File("${tempDir.path}/coolfile.wav");
    await file.writeAsBytes(asset.buffer.asUint8List());

    // open up sound player
    final audioPlayer = FlutterSoundPlayer();
    await audioPlayer.openPlayer();

    // load audio file and play
    await audioPlayer.startPlayer(
        fromURI: "${tempDir.path}/coolfile.wav",
        whenFinished: () => audioPlayer.closePlayer());
  }

  // adapted from stft.run
  void customSTFT(
      List<double> input, Function(Float64x2List) reportChunk, int frameLength,
      [int chunkStride = 0, int fftLength = 512, Float64List? win]) {
    final customfft = FFT(fftLength);
    var chunk = Float64x2List(fftLength);
    var chunk2 = Float64List(fftLength);
    if (chunkStride <= 0) chunkStride = fftLength;
    for (int i = 0;; i += chunkStride) {
      final i2 = i + frameLength;
      if (i2 > input.length) {
        int j = 0;
        final stop = input.length - i;
        for (; j < stop; ++j) {
          chunk[j] = Float64x2(input[i + j], 0);
          chunk2[j] = input[i + j];
        }
        for (; j < frameLength; ++j) {
          chunk[j] = Float64x2.zero();
          chunk2[j] = 0;
        }
      } else {
        for (int j = 0; j < frameLength; ++j) {
          chunk[j] = Float64x2(input[i + j], 0);
          chunk2[j] = input[i + j];
        }
      }
      // apply our window on first frameLength elements instead of entire array
      if (win != null) {
        // TODO only assign to first frameLength elements
        // var padding = Float64x2List(112);
        // chunk =
        chunk2.setRange(0, frameLength,
            win.applyWindowReal(chunk2.sublist(0, frameLength)));

        // we use a scidart hanning here because it gives us the option of a periodic window
        var wind = hann(frameLength, sym: false);
        Float64List choochoo = Float64List(frameLength);
        for (int i = 0; i < frameLength; i++) {
          choochoo[i] = wind[i];
        }
        // WE BELIEVE THIS IS A SYMMETRIC WINDOW, WE NEED PERIODIC
        var windowRes = win.applyWindow(chunk.sublist(0, frameLength));

        // TODO CLEAN THIS UP, ALSO LOOK INTO INACCURACIES WITH THE WINDOWING, AND POTENTIALLY THE FFT FUNCTION
        chunk.setRange(0, frameLength,
            choochoo.applyWindow(chunk.sublist(0, frameLength)));
      }

      //chunk = win?.applyWindow(chunk.sublist(0, frameLength))?? chunk;
      // add more zero padding

      // chunk = win?.applyWindow(chunk); //windowing function
      // pad our zeroes here instead of before windowing
      customfft.inPlaceFft(chunk);
      Float64x2List hunk = customfft.realFft(chunk2);
      //reportChunk(chunk);
      reportChunk(hunk);
      if (i2 >= input.length) {
        break;
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: <Widget>[
        const Text('LID Inference'),
        ElevatedButton(
            style: ButtonStyle(
              backgroundColor: MaterialStatePropertyAll<Color>(Colors.green),
            ),
            onPressed: playAudio,
            child: const Text('Play Audio')),
        ElevatedButton(
            style: ButtonStyle(
              backgroundColor: MaterialStatePropertyAll<Color>(Colors.pink),
            ),
            onPressed: identifyAudio,
            child: const Text('Identify Language!')),
        Text(
          "Identified Language: $identifiedLanguage",
          style: TextStyle(
            fontSize: 24,
            fontWeight: FontWeight.bold,
          ),
        ),
      ],
    );
  }
}
