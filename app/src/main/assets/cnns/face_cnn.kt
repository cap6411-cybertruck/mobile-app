import org.pytorch.nn as nn
import org.pytorch.Tensor
import org.pytorch.functional as F
import org.pytorch.vision.transforms.functional as TF
import org.pytorch.vision.transforms as v2
// import org.opencv
import org.pytorch.vision.transforms.PILToTensor
import org.pytorch.vision.models.alexnet
import org.pytorch.vision.models.mobilenet_v3_small
import org.pytorch.vision.models.timm

import cnn.hands_cnn.getTransforms
import cnn.hands_cnn.visualizeRoi

class FaceCNN(args: Any, numClasses: Int = 10) : nn.Module() {
    private val model: Any = timm.createModel("xception", true, numClasses)

    override fun forward(x: Tensor): Tensor {
        return model(x)
    }
}

fun extractFaceDetections(images: List<Any>, results: Any, trainMode: Any): Any {
    val outputs = mutableListOf<Any>()
    val device = "cuda"

    val (resize, transform) = getTransforms("face", trainMode)

    for ((imageIdx, r) in results.withIndex()) {
        val boxes = r.boxes
        try {
            val bbIndex = boxes.conf.argmax()
            val (x, y, x2, y2) = boxes[bbIndex].xyxy.squeeze().toList()
            val (xInt, yInt, x2Int, y2Int) = listOf(x.toInt(), y.toInt(), x2.toInt(), y2.toInt())

            val box = r.origImg[yInt..y2Int, xInt..x2Int]
            val boxImage = Image.fromArray(box[..., ::-1]) //TODO: conv to opencv 
            val boxImageTensor = resize(PILToTensor(boxImage)).to(device)

            val image = images[imageIdx]
            val origImage = resize(image)

            val stackedImage = transform(torch.cat(origImage, boxImageTensor, 1))
            val stackedImageTensor = stackedImage.to(device)
            // visualizeRoi(stackedImage)
            outputs.add(stackedImageTensor)
        } catch (e: Exception) {
            val blankImage = Image.new("RGB", 299, 299) // TODO: change to opencv
            val blankTensor = transform(blankImage).to(device)
            outputs.add(blankTensor)
        }
    }

    val stackedOutputs = torch.stack(outputs)
    return stackedOutputs.to(device)
}
