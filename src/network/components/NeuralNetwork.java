package src.network.components;

import java.util.Random;

import src.network.config.NetworkConfig;
import src.network.data.DataSet;
import src.network.persist.LayersSnapshot;
import src.network.persist.NetworkSnapshot;
import src.network.stats.NetworkTracker;

public class NeuralNetwork {
    String name;
    NetworkConfig config;
    boolean isTrained;

    Layer[] hiddenLayers;
    Layer outputLayer;

    NeuralNetwork() {}

    public NeuralNetwork(String name, NetworkConfig config) {
        this.name = name;
        this.config = config;
        //this.track = new NetworkTracking(config.outputSize());

        this.hiddenLayers = new Layer[config.hiddenLayerSizes().length];
        int prevLayerSize = config.inputSize();
        for(int i=0; i<config.hiddenLayerSizes().length; i++){
            this.hiddenLayers[i] = new Layer(config.hiddenLayerSizes()[i], prevLayerSize, ActivFunction.RELU);
            prevLayerSize = config.hiddenLayerSizes()[i];
        }

        this.outputLayer = new Layer(config.outputSize(), prevLayerSize, ActivFunction.SIGMOID);
        this.isTrained = false;
    }

    private static double[] feedForward(NeuralNetwork network, double[] input) {
        double[] prevActivs = input;
        for(int i=0; i<network.hiddenLayers.length; i++) {
            prevActivs = network.hiddenLayers[i].computeActivations(prevActivs);
        }
        return network.outputLayer.computeActivations(prevActivs);
    }

    private static void backPropagate(NeuralNetwork network, double[] input, double[] classLabels) {
        // Output layer
        for(int i=0; i<network.outputLayer.size; i++){
            network.outputLayer.errors[i] = (network.outputLayer.activations[i]-classLabels[i])*network.outputLayer.activations[i]*(1-network.outputLayer.activations[i]);
            network.outputLayer.biasGrads[i] += network.outputLayer.errors[i];
        }
        
        // print cost and error for debugging
        //if(network.config.debug()){
        //    System.out.println("\n\t\t\tcost = " + computeCost(network.outputLayer.activations, classLabels) + "\n");
        //    System.out.println("\t\t\td" + (network.hiddenLayers.length+1) + " = " + Arrays.toString(network.outputLayer.errors));
        //}
        
        // move backwards through each hidden layer from the output layer
        Layer nextLayer = network.outputLayer;
        for(int l=network.hiddenLayers.length-1; l>=0; l--){
            for(int i=0; i<network.hiddenLayers[l].size; i++){
                double wtdSumErr = 0.f;
                for(int j=0; j<nextLayer.errors.length; j++){
                    nextLayer.weightGrads[j][i] += nextLayer.errors[j]*network.hiddenLayers[l].activations[i];
                    wtdSumErr += nextLayer.weights[j][i]*nextLayer.errors[j];
                }
                network.hiddenLayers[l].errors[i] = wtdSumErr*network.hiddenLayers[l].activations[i]*(1-network.hiddenLayers[l].activations[i]);
                network.hiddenLayers[l].biasGrads[i] += network.hiddenLayers[l].errors[i];
            }
            
            //if(network.config.debug()) System.out.println("\t\t\td" + (l+1) + " = " + Arrays.toString(network.hiddenLayers[l].errors));
            
            nextLayer = network.hiddenLayers[l];
        }

        // finally, move back through the 1st hidden layer to the input
        for(int i=0; i<input.length; i++){
            for(int j=0; j<nextLayer.size; j++) nextLayer.weightGrads[j][i] += nextLayer.errors[j]*input[i];
        }
    }

    /* TODO: Referenced in `train()` but not yet implemented */
    private static void printClassifyAccuracy(int[] actualCorrect, int[] expectedCorrect) {}

    /* TODO: Referenced in `train()` but not yet implemented */
    private static void printWeights(NeuralNetwork network) {}

    private static void updateWeightsFromGrads(NeuralNetwork network) {
        double c = network.config.learnRate()/(double)network.config.batchSize();
        for(int l=0; l<network.hiddenLayers.length; l++){
            for(int i=0; i<network.hiddenLayers[l].size; i++){
                for(int j=0; j<network.hiddenLayers[l].prevLayerSize; j++) network.hiddenLayers[l].weights[i][j] -= c*network.hiddenLayers[l].weightGrads[i][j];
                network.hiddenLayers[l].biases[i] -= c*network.hiddenLayers[l].biasGrads[i];
            }
        }

        for(int i=0; i<network.outputLayer.size; i++){
            for(int j=0; j<network.outputLayer.prevLayerSize; j++) network.outputLayer.weights[i][j] -= c*network.outputLayer.weightGrads[i][j];
            network.outputLayer.biases[i] -= c*network.outputLayer.biasGrads[i];
        }
    }

    private static void clearCurrGrads(NeuralNetwork network) {
        for(int l=0; l<network.hiddenLayers.length; l++) network.hiddenLayers[l].clearGrads();
        network.outputLayer.clearGrads();
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




    public static NetworkTracker batchTrainNetworkOnDataset(NeuralNetwork network, DataSet dataset) {
        Integer[] randIdxs = dataset.indices();

        NetworkTracker tracker;
        if(network.config.trackTrainOutput())
            tracker = NetworkTracker.newTrainingTracker(network.outputSize(), network.config.epochs());
        else tracker = null;
        //else tracker = NetworkTracking.newTracker(network.outputSize());
        
        for(int epoch=0; epoch < network.config.epochs(); epoch++) {
            shuffleArray(randIdxs);
            clearCurrGrads(network);
            
            // track accuracy for final epoch unless already tracking training output
            if(tracker == null && (epoch+1) == network.config.epochs()) {
                NetworkTracker.newTracker(network.outputSize());
            }

            for(int j=0; j < dataset.size();){
                // verify there are enough inputs for next batch (skip if not full batch)
                if((dataset.size()-j) >= network.config.batchSize()){
                    for(int k=0; k < network.config.batchSize(); j++, k++) {
                        //if(network.config.debug()){
                        //    System.out.println("\t\tTraining Case " + (k+1) + ":");
                        //    System.out.println("\t\t\ta0 = " + Arrays.toString(dataset.inputs()[randIdxs[j]]));
                        //    System.out.println("\t\t\t y = " + Arrays.toString(dataset.labels()[randIdxs[j]]) + "\n");
                        //}

                        feedForward(network, dataset.inputs()[randIdxs[j]]);

                        //if(network.config.debug()){
                        //    for(int l=0; l<network.hiddenLayers.length; l++){
                        //        System.out.println("\t\t\ta" + (l+1) + " = " + Arrays.toString(network.hiddenLayers[l].activations));
                        //    }
                        //    System.out.println("\t\t\ta" + (network.hiddenLayers.length+1) + " = " + Arrays.toString(network.outputLayer.activations));
                        //}

                        if(tracker != null) {
                            if(network.config.trackTrainOutput())
                                trackTrainOutput(tracker, randIdxs[j], epoch,
                                                 dataset.labels()[randIdxs[j]],
                                                 network.outputLayer.activations);
                            else
                                trackOutput(tracker, randIdxs[j], 
                                            dataset.labels()[randIdxs[j]], 
                                            network.outputLayer.activations);
                        }

                        backPropagate(network,
                                    dataset.inputs()[randIdxs[j]], 
                                    dataset.labels()[randIdxs[j]]);
                    }
                    updateWeightsFromGrads(network);

                    //if(network.config.debug()){
                    //    System.out.println("\t Revised Weights");
                    //    printWeights(network);
                    //}
                }
            }

            // Print accuracy data per digit and overall accuracy for the current epoch
            //if(network.config.trackTrainOutput()){
            //    System.out.println(String.format("Epoch %d Accuracy:\n", (i+1)));
            //    printClassifyAccuracy(network.track.epochActualCorrect(), network.track.epochExpectedCorrect());
            //}
        }
    
        network.isTrained = true;
        return tracker;
    }

    public static NetworkTracker testNetworkOnDataset(NeuralNetwork network, DataSet dataset) {
        NetworkTracker tracker = NetworkTracker.newTracker(network.outputSize());

        for(int j=0; j<dataset.size(); j++){
            feedForward(network, dataset.inputs()[j]);
            trackOutput(tracker, j, dataset.labels()[j], network.outputLayer.activations);
        }

        // Print accuracy data per digit and overall accuracy for the current epoch
        //System.out.println("Test Accuracy:\n");
        //printClassifyAccuracy(network.track.totalActualCorrect(), network.track.totalExpectedCorrect());

        return tracker;
    }




    private static void trackOutput(NetworkTracker tracker, int caseIdx, double[] label, double[] activations) {
        int expected = classifyOutput(label);
        int actual = classifyOutput(activations);

        if(actual == expected) {
            tracker.countClassifyCorrect(expected);
        } else {
            tracker.countClassifyMiss(expected, actual, activations, caseIdx);
        }
    }

    private static void trackTrainOutput(NetworkTracker tracker, int caseIdx, int epoch, double[] label, double[] activations) {
        int expected = classifyOutput(label);
        int actual = classifyOutput(activations);

        if(actual == expected) {
            tracker.countTrainClassifyCorrect(expected, epoch);
        } else {
            tracker.countTrainClassifyMiss(expected, actual, activations, caseIdx, epoch);
        }
    }




    public void randomizeWeights() {
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

        this.isTrained = false;
    }




    public String name() {return this.name;}

    public void changeName(String newName) {
        this.name = newName;
    }

    public int inputSize() {return this.config.inputSize();}

    public int outputSize() {return this.config.outputSize();}

    public boolean isTrained() {return this.isTrained;}

    /* TODO: not yet implemented */
    public void adjustInputSize(int newSize) {}
    
    /* TODO: not yet implemented */
    public void adjustOutputSize(int newSize) {}

    public void saveNetwork(String filename) {
        new NetworkSnapshot(this.name, 
                            this.config, 
                            this.hiddenLayers, 
                            this.outputLayer, 
                            this.isTrained)
                    .saveSnapshot(filename);

        /* TODO: verify logic */
        
        // checks for `$WORK_DIR/resources/networks/` directory and create if does not exist
        // then writes snapshot to disk in `/resources/networks/`
    }

    public static NeuralNetwork loadNetwork(String name) {
        NeuralNetwork network = new NeuralNetwork();

        NetworkSnapshot snapshot = NetworkSnapshot.loadSnapshot(name);
        network.name = snapshot.name();
        network.config = snapshot.copyConfig();

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