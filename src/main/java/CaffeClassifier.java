import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.javacv.*;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_dnn.Net;

import static org.bytedeco.opencv.global.opencv_core.CV_32F;
import static org.bytedeco.opencv.global.opencv_core.CV_8UC1;
import static org.bytedeco.opencv.global.opencv_dnn.blobFromImage;
import static org.bytedeco.opencv.global.opencv_dnn.readNetFromCaffe;
import static org.bytedeco.opencv.global.opencv_imgproc.*;


public class CaffeClassifier {
    private static final String PROTO_FILE = "/home/yoga/yolo/caffe/car/deploy.prototxt";
    private static final String CAFFE_MODEL_FILE = "/home/yoga/yolo/caffe/car/googlenet_finetune_web_car_iter_10000.caffemodel";
    private static final Net net;
    static final int IN_WIDTH = 300;
    static final int IN_HEIGHT = 300;
    final float WH_RATIO = (float)IN_WIDTH / IN_HEIGHT;
    static final double IN_SCALE_FACTOR = 0.007843;
    static final double MEAN_VAL = 127.5;
    final double THRESHOLD = 0.2;

    static {
        net = readNetFromCaffe(PROTO_FILE, CAFFE_MODEL_FILE);
    }

    public static void caffeClassifier(String file) {
        FFmpegFrameGrabber grabber = new FFmpegFrameGrabber(file);
        try {
            grabber.start();
            OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
            Mat grabbedImage = converter.convert(grabber.grab());
            int height = grabbedImage.rows();
            int width = grabbedImage.cols();
            CanvasFrame canvasFrame = new CanvasFrame("Screen Capture Caffe");
            canvasFrame.setCanvasSize(width, height);
            cvtColor(grabbedImage, grabbedImage, COLOR_RGBA2RGB);
            Mat blob = blobFromImage(grabbedImage, IN_SCALE_FACTOR, new Size(IN_WIDTH, IN_HEIGHT), new Scalar(MEAN_VAL,MEAN_VAL), /*swapRB*/false, /*crop*/false,0);
            net.setInput(blob);

            while (canvasFrame.isVisible() && grabber.grab() != null) {
                Mat detections = net.forward();
                System.out.println(detections.size());
                canvasFrame.showImage(converter.convert(grabbedImage));
            }
            grabber.stop();
            canvasFrame.dispose();
        } catch (FrameGrabber.Exception ex) {
            ex.printStackTrace();
        }
    }
}
