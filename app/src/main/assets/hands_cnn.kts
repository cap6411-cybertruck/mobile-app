package com.example.myapplication

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.content.res.AssetManager
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.*
import androidx.compose.material3.TopAppBarDefaults.smallTopAppBarColors
import androidx.compose.runtime.Composable
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.myapplication.ui.theme.MyApplicationTheme
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.Arrays
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import org.pytorch.vision.transforms as v2
import org.pytorch.Tensor
import org.pytorch.torchvision.transforms as v2
import org.pytorch.torchvision.transforms.functional as F
import org.pytorch.torchvision.transforms.transforms as transforms
import org.pytorch.torchvision.transforms.transforms.Compose as Compose
import org.pytorch.torchvision.transforms.transforms.Resize as Resize
import org.pytorch.torchvision.transforms.transforms.ToPILImage as ToPILImage
import org.pytorch.torchvision.transforms.transforms.ToTensor as ToTensor
import org.pytorch.torchvision.transforms.transforms.Normalize as Normalize
import org.pytorch.torchvision.transforms.transforms.RandomHorizontalFlip as RandomHorizontalFlip
import org.pytorch.torchvision.transforms.transforms.RandomPerspective as RandomPerspective
import org.pytorch.torchvision.transforms.transforms.ToImage as ToImage
import org.pytorch.torchvision.transforms.transforms.ToDtype as ToDtype


fun getTransforms(modelType: String = "hands_vgg", trainMode: Boolean = true): Pair<Resize, Compose> {
    val resize: Resize = if (modelType == "hands_vgg") {
        Resize(224, 224)
    } else if (modelType == "face") {
        Resize(299, 299)
    } else {
        throw IllegalArgumentException("Invalid model type")
    }
// here
    val transform: Compose = 
        Compose(
            listOf(
                ToPILImage(),
                resize,
                ToImage(),
                ToDtype(Tensor.Type.FLOAT32, true),
                Normalize(listOf(0.3879, 0.3879, 0.3879), listOf(0.3001, 0.3001, 0.3001))
            )
        )

    return Pair(resize, transform)
    }

fun val(
    model: Any,
    detector: Any,
    device: Any,
    testLoader: Any,
    criterion: Any,
    epoch: Int,
    modelName: String = "hands_vgg"
): Pair<Double, Double> {
    val losses = mutableListOf<Double>()
    var total = 0
    var correct = 0

    testLoader.forEach { sample ->
        val (data, target) = sample
        data.to(device)
        target.to(device)
        val output = model(data)

        val loss = criterion(output, target)
        losses.add(loss.item().toDouble())

        val pred = output.argmax(1, true)
        total += target.size(0)
        correct += pred.eq(target.view_as(pred)).sum().item().toInt()
    }

    val testLoss = losses.average()
    val accuracy = (correct.toDouble() / total) * 100.0

    println("Validation at epoch $epoch")
    println("Average loss: %.4f, Accuracy: %d/%d (%.2f%)".format(testLoss, correct, total, accuracy))
    return Pair(testLoss, accuracy)
}

// class MainActivity : ComponentActivity() {
// }

@OptIn(HandsCNNApi::class)
override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)

    val bitmap = BitmapFactory.decodeStream(assets.open("other.jpg"))

    val detector = LiteModuleLoader.load(assetFilePath(this, "app/src/main/assets/best_hands_detector.ptl"))
    val hands_cnn = LiteModuleLoader.load(assetFilePath(this, "app/src/main/assets/Hands_CNN.ptl"))
    hands_cnn.eval()

    val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
        bitmap,
        TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB
    )

    detections = detector(inputTensor)
    rois = extractHandsDetection(bitmap, detections, null, "hands_vgg16", true, false)
    val outputTensor = hands_model.forward(IValue.from(rois)).toTensor()

    }

    /**
     * Concatenates the boxes and performs resizing and transformation on the input image.
     * @param result The result of the detection model
     * @param image The input image
     * @param numBoxes The number of boxes in the result
     * @param resize The resize function
     * @param transform The transformation function
     * @param useOrigImg Flag to indicate whether to use the original image
     * @return The stacked and transformed image
     */
    override fun concatenateBoxes(result: DetectionResult, image: Image, numBoxes: Int, resize: (Image) -> Image, transform: (Image) -> Image, useOrigImg: Boolean): Image {
        val rois = mutableListOf<Image>()

        val topIdxs = if (numBoxes > 2) {
            val (_, topIdxs) = result.boxes.conf.topk(2)
            topIdxs
        } else {
            null
        }

        topIdxs?.let {
            for ((box, cls) in result.boxes.xyxy[it].squeeze(0).zip(result.boxes.cls[it].squeeze(0))) {
                val (xMin, yMin, xMax, yMax) = box.toInt()
                val roi = image[:, yMin..yMax, xMin..xMax]
                rois.add(roi)
            }
        }

        if (topIdxs != null && cls == 0) rois.reverse()

        val stackedRois = if (numBoxes > 1) {
            val transformedRois = rois.map { resize(it) }
            resize(torch.cat(transformedRois, dim = 2))
        } else {
            resize(rois.first())
        }

        rois.clear()

        val stackedImage = if (useOrigImg) {
            val origImage = resize(image)
            transform(torch.cat(origImage, stackedRois, dim = 1))
        } else {
            transform(stackedRois)
        }

        return stackedImage
    }

    /**
     * Extracts hands detection from the images and results.
     * @param images The input images
     * @param results The detection results
     * @param target The target data
     * @param modelName The name of the model
     * @param useOrigImg Flag to indicate whether to use the original image
     * @param trainMode Flag to indicate whether the model is in training mode
     * @return The extracted data and target
     */
    override fun extractHandsDetection(images: List<Image>, results: List<DetectionResult>, target: List<Target>?, modelName: String, useOrigImg: Boolean = true, trainMode: Boolean = true): Pair<Tensor, Tensor>? {
        val data = mutableListOf<Image>()
        val targetList = mutableListOf<Target>()
        val (resize, transform) = getTransforms(modelName, trainMode)

        for ((imgIdx, result) in results.withIndex()) {
            val numBoxes = result.boxes.size

            if (numBoxes == 0 || numBoxes > 4) {
                if (target == null) {
                    data.add(transform(images[imgIdx]))
                }
                continue
            }

            val stackedRois = concatenateBoxes(result, images[imgIdx], numBoxes, resize, transform, useOrigImg)
            data.add(stackedRois)

            if (target != null) {
                targetList.add(target[imgIdx])
            }
        }

        val tensorData = data.stack().toDevice("cuda")

        return if (target != null) {
            val tensorTarget = targetList.stack()
            require(tensorData.size(0) == tensorTarget.size(0)) { "Batch size of data must be equal to target length." }
            tensorData to tensorTarget
        } else {
            tensorData
        }
    }
