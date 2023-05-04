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
import 'package:flutter_sound/flutter_sound.dart';
import 'package:permission_handler/permission_handler.dart';
import 'audio_processing.dart';

void main() {
  runApp(const MainApp());
}

class MainApp extends StatelessWidget {
  const MainApp({super.key});

  Future<void> btnListener()
  async {
    final tempDir = await getTemporaryDirectory();

    // final metadata = await FFprobeKit.getMediaInformation('${tempDir.path}/sample.mp3');
    // final mediaInfo = metadata.getMediaInformation()?.getAllProperties()!['streams'][0];
    // final int sampleRate = int.parse(mediaInfo['sample_rate']);
    // print(sampleRate);

    if(!(await Permission.microphone.request().isGranted)) {
      print("No permissions!");
      return;
    }

    // final FlutterSoundRecorder recorder = FlutterSoundRecorder();
    //
    // await recorder.openRecorder();
    // //
    // await recorder.startRecorder(
    //     codec: Codec.defaultCodec,
    //     toFile: "${tempDir.path}/recordings/myrecording",
    //     sampleRate: 16000
    // );
    // //
    // await Future.delayed(Duration(seconds: 8));
    //
    // final url = await recorder.stopRecorder();
    // await recorder.closeRecorder();
    // print(url);

    // try and record audio
    //final audio = await readMP3('sample.mp3', resample: 16000);
    // final FlutterSoundPlayer audioplayer = FlutterSoundPlayer();
    // await audioplayer.openPlayer();
    // //
    // final duration = await audioplayer.startPlayer(fromURI: "${tempDir.path}/sample.mp3", whenFinished: () async {
    //    print("finished");
    //    await audioplayer.closePlayer();
    // });
    // still need to convert this file
    // output path for raw data
    final String outputPath = "${tempDir.path}/output/output.raw";

    // ffmpeg command
    // -i: this is the input path
    // -f: this is the output type we want (float 32, little-endian)
    // -acodec: the codec we want to use to format the data, we want pcm data
    final String command = "-i ${tempDir.path}/sample.mp3 -f f32le -acodec pcm_f32le $outputPath";

    // execute the command
    final ffmpegSession = await FFmpegKit.executeAsync(command);
    final returnCode = await ffmpegSession.getReturnCode().then((value) => value?.getValue());

    if(returnCode != 1) {
      throw Exception('FFMPEG Conversion Failed!\n ${await ffmpegSession.getOutput()}');
    }
    //final pcmData = await getPCMData(outputPath);
    //print(pcmData);
    //print(pcmData.length);

    // why is this blatently half the length of the other one?

    //
    // print(audioplayer.isPlaying);
    //await audioplayer.closePlayer();
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
