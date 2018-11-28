package info.magnolia.ai.similarity;

import static java.util.stream.Collectors.toList;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.Arrays;
import java.util.Collection;
import java.util.stream.IntStream;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class Trainer {

    private static final float[] POSITIVE_OUTPUT = {1};
    private static final float[] NEGATIVE_OUTPUT = {0};
    private static final int NUM_OUTPUTS = POSITIVE_OUTPUT.length;

    private final ComputationGraph pretrainedNet;
    private final ComputationGraph transferGraph;

    private final Collection<INDArray> positiveInputs;
    private final Collection<INDArray> negativeInputs;

    public Trainer(final String[] positives, final String[] negatives) throws IOException {
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);

        pretrainedNet = (ComputationGraph) VGG16.builder().build().initPretrained(PretrainedType.IMAGENET);

        final FineTuneConfiguration fineTuneConfiguration = new FineTuneConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(0.01d / (positives.length + negatives.length), 0.1))
                .build();
        transferGraph = new TransferLearning.GraphBuilder(pretrainedNet)
                .fineTuneConfiguration(fineTuneConfiguration)
                .setFeatureExtractor("fc2") // freeze this and below
//                .removeVertexAndConnections("predictions") // XXX: Maybe this?
                .removeVertexKeepConnections("predictions")
                .addLayer("predictions",
                        new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                                .nIn(4096).nOut(NUM_OUTPUTS) // santa hat or not santa hat
                                .weightInit(WeightInit.ZERO)
                                .activation(Activation.SIGMOID)
                                .build(), "fc2")
                .build();
        System.out.println("Transfer model:");
        System.out.println(transferGraph.summary());

        System.out.println("Encoding positive samples...");
        this.positiveInputs = Arrays.stream(positives).map(this::encodeImage).collect(toList());
        System.out.println("Encoding negative samples...");
        this.negativeInputs = Arrays.stream(negatives).map(this::encodeImage).collect(toList());

        System.out.println("Output layer:");
        System.out.println(transferGraph.getOutputLayer(0).params());
    }

    private INDArray encodeImage(final String filePath) {
        final NativeImageLoader imageLoader = new NativeImageLoader(224, 224, 3);
        final VGG16ImagePreProcessor scaler = new VGG16ImagePreProcessor();
        try {
            final File dir = new File(App.class.getResource(".").toURI());
            final INDArray imageArray = imageLoader.asMatrix(new File(dir, filePath));
            scaler.transform(imageArray);
            return imageArray;
        } catch (IOException | URISyntaxException e) {
            throw new RuntimeException(e);
        }
    }

    public void train(final int iterations) {
        // FIXME: Training sometimes "kills" weights, i.e. ends up with NaNs. Looks like arithmetic underflow, but why?
        IntStream.range(0, iterations).forEach(i -> this.runIteration());
    }

    private void runIteration() {
        System.out.println("Running iteration...");
        this.positiveInputs.forEach(input -> {
            final DataSet dataSet = new DataSet(input, new NDArray(POSITIVE_OUTPUT));
            System.out.print("☑");
            this.transferGraph.fit(dataSet);
        });
        this.negativeInputs.forEach(input -> {
            System.out.print("☐");
            final DataSet dataSet = new DataSet(input, new NDArray(NEGATIVE_OUTPUT));
            this.transferGraph.fit(dataSet);
        });
        System.out.println();

        System.out.println("Output layer:");
        System.out.println(transferGraph.getOutputLayer(0).params());
    }

    public double check(final String imagePath) {
        final INDArray input = encodeImage(imagePath);
        // FIXME: Prevent normalization! otherwise, always [1]
        final INDArray result = this.transferGraph.outputSingle(input);

        return result.getFloat(0);
    }

    public INDArray checkWithPretrained(final String imagePath) {
        return this.pretrainedNet.outputSingle(encodeImage(imagePath));
    }
}
