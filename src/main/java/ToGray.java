import org.bytedeco.javacv.*;
import org.bytedeco.opencv.opencv_core.Mat;

import static org.bytedeco.opencv.global.opencv_core.CV_8UC1;
import static org.bytedeco.opencv.global.opencv_imgproc.CV_BGR2GRAY;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;

public class ToGray {
    public static void toGray(String file){
        FFmpegFrameGrabber grabber = new FFmpegFrameGrabber(file);
        try{
            grabber.start();
            OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
            Mat grabbedImage = converter.convert(grabber.grab());
            int height = grabbedImage.rows();
            int width = grabbedImage.cols();
            Mat grayImage = new Mat(height, width, CV_8UC1);
            CanvasFrame frame = new CanvasFrame("Screen Capture Gray");
            frame.setCanvasSize(width,height);
            while (frame.isVisible() && grabber.grab() !=null) {
                cvtColor(grabbedImage, grayImage, CV_BGR2GRAY);
                frame.showImage(converter.convert(grayImage));
            }
            frame.dispose();
            grabber.stop();
        }catch (FrameGrabber.Exception ex){
            ex.printStackTrace();
        }
    }
}
