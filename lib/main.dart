import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'dart:math';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:io';
import 'package:ffmpeg_kit_flutter_audio/ffmpeg_kit.dart';
import 'package:ffmpeg_kit_flutter_audio/ffprobe_kit.dart';
import 'package:fftea/fftea.dart';
import 'package:csv/csv.dart';

void main() {
  runApp(const MainApp());
}

class MainApp extends StatelessWidget {
  const MainApp({super.key});

  // adapted from scipy.signal.resample
  // resamples given audio into new length
  Float64List? resampleAudio(List<double> x, int num) {
    // initialize fft
    final fft1 = FFT(x.length);
    final Float64x2List X;
    // fft operations on axis 0
    // to do a complex array need to convert to complexArray dtype
    X = fft1.realFft(x);

    final Nx = x.shape[0];
    final N = min(num, Nx);
    var nyq = (N / 2).floor() + 1;

    // initializes list full of zeroes of length nyq
    Float64x2List Y = Float64x2List(nyq);
    // Float64x2List Y = Float64x2List(N); // old

    Y.setRange(0, nyq, X.sublist(0, nyq).toList()); // copy X[:nyq] to Y[:nyq]
    print(num);

    if(N % 2 == 0) {
      if (num < Nx) {
        var sl1 = (N / 2).floor();
        //var sl2 = (N / 2).floor() + 1;
        Y[sl1] = Y[sl1].scale(2.0);
        //Y.setRange(sl1, sl2, Y.sublist(sl1, sl2).map((e) => e.scale(2.0)));
      } else if (num > Nx) {
        var sl1 = (N / 2).floor();
        Y[sl1] = Y[sl1].scale(0.5);
      }
    }

    // padding zeros
    Float64x2List newY = Float64x2List(num); // len 140000
    newY.setRange(0, Y.length, Y.sublist(0)); // Y > newY,

    final fft2 = FFT(Y.length);
    // inverse transforms
    //print(Y);
    //fft2.realInverseFft(newY);
    //print(newY);
    //Float64List y = customInverseFft(newY, fft2, nyq);
    final complexArray = Float64x2List.fromList([
      Float64x2(1.0, 0.0),
      Float64x2(0.0, 0.0),
      Float64x2(1.0, 0.0),
      Float64x2(0.0, 0.0),
    ]);
    Float64List y = customInverseFft(Y, FFT(Y.length));
    print(y);
    //Float64List y = customInverseFft(newY, FFT(newY.length));
    //print(y);
    //print(y);
    //print(newY[75000]);

    //y.forEach((element) { })

    y = Float64List.fromList(y.map((e) => e *= (num / Nx)).toList());

    return y;
  }

  Float64List customInverseFft(Float64x2List complexArray, FFT f) {
    f.inPlaceFft(complexArray);

    final len = complexArray.length;
    final scale = 1 / (len.toDouble());
    final r = Float64List(len);

    r[0] = complexArray[0].x * scale;
    if (len <= 1) return r;
    for (int i = 1; i < len; ++i) {
      r[i] = complexArray[len - i].x * scale;
    }
    return r;
  }

  Future<Map<String, dynamic>> readMP3(inputPath, {resample = 16000}) async {
    // saving to tempdir
    final tempDir = await getTemporaryDirectory();

    final metadata = await FFprobeKit.getMediaInformation('${tempDir.path}/$inputPath');
    final mediaInfo = metadata.getMediaInformation()?.getAllProperties()!['streams'][0];
    final int sampleRate = int.parse(mediaInfo['sample_rate']);

    // output path for raw data
    final String outputPath = "${tempDir.path}/output.raw";

    // ffmpeg command
    // -i: this is the input path
    // -f: this is the output type we want (float 32, little-endian)
    // -acodec: the codec we want to use to format the data, we want pcm data
    // final String command = "-i ${tempDir.path}/$inputPath -f f32le -acodec pcm_f32le $outputPath";
    //
    // // execute the command
    // final ffmpegSession = await FFmpegKit.executeAsync(command);
    // final returnCode = await ffmpegSession.getReturnCode().then((value) => value?.getValue());
    //
    // if(returnCode != 1) {
    //   throw Exception('FFMPEG Conversion Failed!\n ${await ffmpegSession.getOutput()}');
    // }

    // grab the raw data
    //final pcmData = await getPCMData(outputPath);
    // REMOVE THESE TWO LINES
    final bytes = await rootBundle.load('assets/output.raw');
    final pcmData = Float32List.view(bytes.buffer);

    // calc the new length based on resample rate
    final newLength = (pcmData.length * resample / sampleRate).round();

    // finally, get the signal
    final signal = resampleAudio(pcmData.toList(), newLength);

    Map<String, dynamic> audio = {"signal": signal,"rate": resample};
    return audio;
  }

  Future<void> btnListener()
  async {
    final audio = await readMP3('sample.mp3', resample: 16000);
    //print(audio['signal']?.length);

    // need to remove silence on signal

    // then create spectrogram

    // then we're ready.... easy right?

    //print(pcmData);

    final interpreter = await Interpreter.fromAsset('model.tflite');

    // grab shapes
    // TODO figure out resizing
    // shape: (num_batch, num_frames, 40)
    interpreter.resizeInputTensor(0, [1, 1, 40]);
    var inputShape = interpreter.getInputTensor(0).shape;
    var outputShape = interpreter.getOutputTensor(0).shape;
    // set shapes on dummy data and output tensor
    var input = List.filled(40, Random().nextDouble()).reshape(inputShape);
    var output = List.filled(1*4, 0).reshape(outputShape);
    // run interpreter
    interpreter.run(input, output);
    print(output);
  }

  Future<Float32List> getPCMData(String path) async {
    final file = File(path);
    if(!await file.exists()) {
      throw Exception('File not found at $path');
    }
    final bytes = await file.readAsBytes();
    final buffer = Float32List.view(bytes.buffer);

    // delete file
    file.delete();
    return buffer;
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'LID Inference',
      home: Scaffold(
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: <Widget> [
              const Text('LID Inference'),
              ElevatedButton(onPressed: btnListener, child: const Text('Press me!'))
            ],
          ),
        ),
      ),
    );
  }
}
