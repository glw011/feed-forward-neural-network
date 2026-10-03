package src.network.components;

import java.util.Arrays;
import java.util.Random;

import src.network.config.NetworkConfig;
import src.network.data.DataSet;
import src.network.persist.LayersSnapshot;
import src.network.persist.NetworkSnapshot;
import src.network.tracking.NetworkTracking;
import src.network.util.NNMath;

public class NeuralNetwork {
    String name;
    NetworkConfig config;
    NetworkTracking track;

    Layer[] hiddenLayers;
    Layer outputLayer;

    NeuralNetwork() {}

    public NeuralNetwork(String name, NetworkConfig config) {
        this.name = name;
        this.config = config;
        this.track = new NetworkTracking(config.outputSize());

        this.hiddenLayers = new Layer[config.hiddenLayerSizes().length];
        int prevLayerSize = config.inputSize();
        for(int i=0; i<config.hiddenLayerSizes().length; i++){
            this.hiddenLayers[i] = new Layer(config.hiddenLayerSizes()[i], prevLayerSize, ActivFunction.RELU);
            prevLayerSize = config.hiddenLayerSizes()[i];
        }

        this.outputLayer = new Layer(config.outputSize(), prevLayerSize, ActivFunction.SIGMOID);
    }

    private double[] deprecated_feedForward(double[] input) {
        double[] curr = {};
        double[] prevOut = input;
        for(int i=0; i<this.hiddenLayers.length; i++){
            if(curr.length != this.hiddenLayers[i].size) curr = new double[this.hiddenLayers[i].size];
            for(int j=0; j<this.hiddenLayers[i].size; j++){
                curr[j] = NNMath.vDot(this.hiddenLayers[i].weights[j], prevOut);
            }
            this.hiddenLayers[i].activations = NNMath.sigmoid(NNMath.vAdd(curr, this.hiddenLayers[i].biases));
            prevOut = this.hiddenLayers[i].activations;
        }

        if(curr.length != this.outputLayer.size) curr = new double[this.outputLayer.size];
        for(int i=0; i<outputLayer.size; i++){
            curr[i] = NNMath.vDot(this.outputLayer.weights[i], prevOut);
        }
        this.outputLayer.activations = NNMath.sigmoid(NNMath.vAdd(curr, this.outputLayer.biases));
        return this.outputLayer.activations;
    }

    private double[] feedForward(double[] input) {
        double[] prevActivs = input;
        for(int i=0; i<this.hiddenLayers.length; i++) {
            prevActivs = this.hiddenLayers[i].computeActivations(prevActivs);
        }
        return this.outputLayer.computeActivations(prevActivs);
    }

    private void backPropagate(double[] input, double[] classLabels) {
        // Output layer
        for(int i=0; i<this.outputLayer.size; i++){
            this.outputLayer.errors[i] = (this.outputLayer.activations[i]-classLabels[i])*this.outputLayer.activations[i]*(1-this.outputLayer.activations[i]);
            this.outputLayer.biasGrads[i] += this.outputLayer.errors[i];
        }
        
        // print cost and error for debugging
        if(this.config.debug()){
            System.out.println("\n\t\t\tcost = " + computeCost(this.outputLayer.activations, classLabels) + "\n");
            System.out.println("\t\t\td" + (this.hiddenLayers.length+1) + " = " + Arrays.toString(this.outputLayer.errors));
        }
        
        // move backwards through each hidden layer from the output layer
        Layer nextLayer = this.outputLayer;
        for(int l=this.hiddenLayers.length-1; l>=0; l--){
            for(int i=0; i<this.hiddenLayers[l].size; i++){
                double wtdSumErr = 0.f;
                for(int j=0; j<nextLayer.errors.length; j++){
                    nextLayer.weightGrads[j][i] += nextLayer.errors[j]*this.hiddenLayers[l].activations[i];
                    wtdSumErr += nextLayer.weights[j][i]*nextLayer.errors[j];
                }
                this.hiddenLayers[l].errors[i] = wtdSumErr*this.hiddenLayers[l].activations[i]*(1-this.hiddenLayers[l].activations[i]);
                this.hiddenLayers[l].biasGrads[i] += this.hiddenLayers[l].errors[i];
            }
            if(this.config.debug()) System.out.println("\t\t\td" + (l+1) + " = " + Arrays.toString(this.hiddenLayers[l].errors));
            nextLayer = this.hiddenLayers[l];
        }

        // finally, move back through the 1st hidden layer to the input
        for(int i=0; i<input.length; i++){
            for(int j=0; j<nextLayer.size; j++) nextLayer.weightGrads[j][i] += nextLayer.errors[j]*input[i];
        }
    }

    private void train(DataSet dataset) {
        int setSize = dataset.size();
        Integer[] randIdxs = dataset.indices();
        
        for(int i=0; i<this.config.epochs(); i++){
            shuffleArray(randIdxs);
            clearPrevGrads();
            if(this.config.trackOutput()){
                this.track.clearEpochCounts();
            }

            for(int j=0; j<setSize;){
                // verify there are enough inputs for next batch (skip if not full batch)
                if((setSize-j) >= this.config.batchSize()){
                    for(int k=0; k<this.config.batchSize(); j++, k++){
                        if(this.config.debug()){
                            System.out.println("\t\tTraining Case " + (k+1) + ":");
                            System.out.println("\t\t\ta0 = " + Arrays.toString(dataset.inputs()[randIdxs[j]]));
                            System.out.println("\t\t\t y = " + Arrays.toString(dataset.labels()[randIdxs[j]]) + "\n");
                        }

                        feedForward(dataset.inputs()[randIdxs[j]]);

                        if(this.config.debug()){
                            for(int l=0; l<this.hiddenLayers.length; l++){
                                System.out.println("\t\t\ta" + (l+1) + " = " + Arrays.toString(this.hiddenLayers[l].activations));
                            }
                            System.out.println("\t\t\ta" + (this.hiddenLayers.length+1) + " = " + Arrays.toString(this.outputLayer.activations));
                        }

                        if(this.config.trackOutput()) updateOutputTracking(
                                                            randIdxs[j],
                                                            dataset.labels()[randIdxs[j]],
                                                            this.outputLayer.activations);

                        backPropagate(
                                dataset.inputs()[randIdxs[j]], 
                                dataset.labels()[randIdxs[j]]);
                    }
                    updateWeights(this.config.batchSize());

                    if(this.config.debug()){
                        System.out.println("\t Revised Weights");
                        printWeights();
                    }
                }
            }

            // Print accuracy data per digit and overall accuracy for the current epoch
            if(this.config.trackOutput()){
                System.out.println(String.format("Epoch %d Accuracy:\n", (i+1)));
                printClassifyAccuracy(this.track.epochActualCorrect(), this.track.epochExpectedCorrect());
            }
        }
    }

    /* TODO: Referenced in `train()` but not yet implemented */
    private void printClassifyAccuracy(int[] actualCorrect, int[] expectedCorrect) {}

    /* TODO: Referenced in `train()` but not yet implemented */
    private void printWeights() {}

    private void randomizeWeights() {
        Random rand = new Random();
        for(int l=0; l<this.hiddenLayers.length; l++){
            for(int i=0; i<this.hiddenLayers[l].size; i++){
                for(int j=0; j<this.hiddenLayers[l].prevLayerSize; j++) this.hiddenLayers[l].weights[i][j] = rand.nextDouble()*2-1;
                this.hiddenLayers[l].biases[i] = rand.nextDouble()*2-1;
            }
        }

        for(int i=0; i<this.outputLayer.size; i++){
            for(int j=0; j<this.outputLayer.prevLayerSize; j++) this.outputLayer.weights[i][j] = rand.nextDouble()*2-1;
            this.outputLayer.biases[i] = rand.nextDouble()*2-1;
        }
    }

    private void updateWeights(int batchSize) {
        double c = this.config.learnRate()/(double)batchSize;
        for(int l=0; l<this.hiddenLayers.length; l++){
            for(int i=0; i<this.hiddenLayers[l].size; i++){
                for(int j=0; j<this.hiddenLayers[l].prevLayerSize; j++) this.hiddenLayers[l].weights[i][j] -= c*this.hiddenLayers[l].weightGrads[i][j];
                this.hiddenLayers[l].biases[i] -= c*this.hiddenLayers[l].biasGrads[i];
            }
        }

        for(int i=0; i<this.outputLayer.size; i++){
            for(int j=0; j<this.outputLayer.prevLayerSize; j++) this.outputLayer.weights[i][j] -= c*this.outputLayer.weightGrads[i][j];
            this.outputLayer.biases[i] -= c*this.outputLayer.biasGrads[i];
        }
    }

    private void clearPrevGrads() {
        for(int l=0; l<this.hiddenLayers.length; l++) this.hiddenLayers[l].clearGrads();
        this.outputLayer.clearGrads();
    }

    /**
     * Shuffles contents of an array using Fisher-Yates algorithm
     * @param <T>
     * @param array
     */
    private static <T> void shuffleArray(T[] array) {
        Random rand = new Random();

        int j; 
        T temp;
        for(int i=array.length; i>0; i--) {
            j = rand.nextInt(i+1);
            temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }

    private void updateOutputTracking(int testCase, double[] label, double[] activations) {
        int expected = classifyOutput(label);
        int actual = classifyOutput(activations);
        
        if(actual == expected){
            this.track.classifyCorrect(expected, actual);
        } else {
            this.track.classifyMiss(expected, actual, activations, testCase);
        }
    }

    private static double computeCost(double[] activations, double[] label) {
        double diffSum = 0.f;
        for(int i=0; i<activations.length; i++) diffSum += Math.pow(label[i]-activations[i], 2);
        return 0.5f*diffSum;
    }

    private static int classifyOutput(double[] activations) {
        int pick = 0;
        for(int i=1; i<activations.length; i++){
            if(activations[i] > activations[pick]) pick = i;
        }
        return pick;
    }

    public String name() {return this.name;}

    public void changeName(String newName) {
        this.name = newName;
    }

    public int inputSize() {return this.config.inputSize();}

    public int outputSize() {return this.config.outputSize();}

    /* TODO: not yet implemented */
    public void adjustInputSize(int newSize) {}
    
    /* TODO: not yet implemented */
    public void adjustOutputSize(int newSize) {}

    public void saveNetwork(String filename) {
        new NetworkSnapshot(this.name, this.config, this.hiddenLayers, this.outputLayer)
                    .saveSnapshot(filename);

        // checks for `$WORK_DIR/resources/networks/` directory and create if does not exist
        // then writes snapshot to disk in `/resources/networks/`
    }

    public static NeuralNetwork loadNetwork(String name) {
        NeuralNetwork network = new NeuralNetwork();

        NetworkSnapshot snapshot = NetworkSnapshot.loadSnapshot(name);
        network.name = snapshot.name();
        network.config = snapshot.copyConfig();
        network.track = new NetworkTracking(network.config.outputSize());

        LayersSnapshot layers = snapshot.layersSnapshot();
        network.hiddenLayers = layers.hiddenLayers();
        network.outputLayer = layers.outputLayer();

        return network;
    }

    public static NetworkContainer defaultMnistNetwork() {
        NeuralNetwork network = new NeuralNetwork("Default MNIST Digit Classifier", NetworkConfig.defaultMnistConfig());
        network.randomizeWeights();

        DataSet trainData = DataSet.defaultMnistTrainingSet();
        DataSet testData = DataSet.defaultMnistTestingSet();

        return new NetworkContainer(network, trainData, testData);
    }
}