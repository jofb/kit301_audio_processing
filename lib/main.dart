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
import 'audio_processing.dart';

void main() {
  runApp(const MainApp());
}

class MainApp extends StatelessWidget {
  const MainApp({super.key});

  Future<void> btnListener()
  async {

    /******************* SPECTROGRAM STUFF *******************/
    // the resampled audio is in the assets folder under filtered_audio.wav

    /******************* MODEL STUFF *******************/
    // final interpreter = await Interpreter.fromAsset('model.tflite');
    //
    // // grab shapes
    // // TODO figure out resizing
    // // shape: (num_batch, num_frames, 40)
    // interpreter.resizeInputTensor(0, [1, 1, 40]);
    // var inputShape = interpreter.getInputTensor(0).shape;
    // var outputShape = interpreter.getOutputTensor(0).shape;
    // // set shapes on dummy data and output tensor
    // var input = List.filled(40, Random().nextDouble()).reshape(inputShape);
    // var output = List.filled(1*4, 0).reshape(outputShape);
    // // run interpreter
    // interpreter.run(input, output);
    // print(output);
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
