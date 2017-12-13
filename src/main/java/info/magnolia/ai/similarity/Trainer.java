package info.magnolia.ai.similarity;

import static java.util.stream.Collectors.toList;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.stream.IntStream;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.model.ResNet50;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class Trainer {

    private static final double[][] POSITIVE_OUTPUT = {{1}};
    private static final double[][] NEGATIVE_OUTPUT = {{0}};

    private final ComputationGraph transferGraph;

    private final Collection<INDArray> positiveInputs;
    private final Collection<INDArray> negativeInputs;

    public Trainer(final String[] positives, final String[] negatives) throws IOException {
        final ComputationGraph pretrainedNet = (ComputationGraph) new ResNet50().initPretrained(PretrainedType.IMAGENET);
        final FineTuneConfiguration fineTuneConfiguration = new FineTuneConfiguration.Builder()
                .learningRate(1d / (positives.length + negatives.length))
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS)
                .build();
        transferGraph = new TransferLearning.GraphBuilder(pretrainedNet)
                .fineTuneConfiguration(fineTuneConfiguration)
                .setFeatureExtractor("fc2") // XXX: What is this?
                .removeVertexAndConnections("predictions") // XXX: What does this do?
                .addLayer("predictions",
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nIn(4096).nOut(1) // santa hat or not santa hat
                                .weightInit(WeightInit.XAVIER)
                                .activation(Activation.SOFTMAX).build(), "fc2")
                .build();

        this.positiveInputs = Arrays.stream(positives).map(this::encodeImage).collect(toList());
        this.negativeInputs = Arrays.stream(negatives).map(this::encodeImage).collect(toList());
    }

    private INDArray encodeImage(final String filePath) {
        final NativeImageLoader imageLoader = new NativeImageLoader(224, 224, 3);
        try {
            return imageLoader.asMatrix(new File(filePath));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public void train(final int iterations) {
        IntStream.range(0, iterations).forEach(i -> this.runIteration());
    }

    private void runIteration() {
        this.positiveInputs.forEach(input -> {
            final DataSet dataSet = new DataSet(input, new NDArray(POSITIVE_OUTPUT));
            this.transferGraph.fit(dataSet);
        });
        this.negativeInputs.forEach(input -> {
            final DataSet dataSet = new DataSet(input, new NDArray(POSITIVE_OUTPUT));
            this.transferGraph.fit(dataSet);
        });
    }
}
