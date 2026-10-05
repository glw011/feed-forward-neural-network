package src.network.config;

public class NetworkConfig {
    boolean debug;               // prints additional network info during training/tests for debugging
    boolean trackTrainOutput;         // track the accuracy stats of individual digits
    boolean showDistributions;   // show distribution for output when running network on testing data

    int inputSize;               // number of nodes in input layer
    int[] hiddenLayerSizes;           // sequential array containing number of nodes in each of the hidden layers
    int outputSize;              // number of nodes in output layer

    int batchSize;               // batch size used for network training
    int epochs;                  // number of training epochs to perform          
    double learnRate;            // learning rate 'eta'
    
    NetworkConfig() {
        this.debug = false;
        this.trackTrainOutput = true;
        this.showDistributions = true;

        this.inputSize = 784;
        this.outputSize = 10;
        int[] hidden = {256, 128, 64, 32};
        this.hiddenLayerSizes = hidden;

        this.batchSize = 10;
        this.epochs = 10;
        this.learnRate = 0.15f;
    }

    public NetworkConfig(int inputSize, int outputSize, int[] hiddenLayerSizes,
                         int batchSize, int epochs, double learnRate) {
        this.debug = false;
        this.trackTrainOutput = true;
        this.showDistributions = true;

        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.hiddenLayerSizes = hiddenLayerSizes;

        setBatchSize(batchSize);
        setEpochs(epochs);
        setLearnRate(learnRate);
    }

    public NetworkConfig(NetworkConfig other) {
        this.debug = other.debug;
        this.trackTrainOutput = other.trackTrainOutput;
        this.showDistributions = other.showDistributions;

        this.inputSize = other.inputSize;
        this.outputSize = other.outputSize;
        this.hiddenLayerSizes = other.hiddenLayerSizes;

        this.batchSize = other.batchSize;
        this.epochs = other.epochs;
        this.learnRate = other.learnRate;
    }

    public static NetworkConfig defaultMnistConfig() {return new NetworkConfig();}

    public void debugOn() {this.debug = true;}
    public void debugOff() {this.debug = false;}

    public void trackTrainOutputOn() {this.trackTrainOutput = true;}
    public void trackTrainOutputOff() {this.trackTrainOutput = false;}

    public void distributionsOn() {this.showDistributions = true;}
    public void distributionsOff() {this.showDistributions = false;}

    public void setBatchSize(int batchSize) {this.batchSize = (batchSize > 0) ? batchSize : 10;}
    public void setEpochs(int epochs) {this.epochs = (epochs > 0) ? epochs : 10;}
    public void setLearnRate(double learnRate) {
        this.learnRate = (learnRate > 0 && learnRate < 1) ? learnRate : 0.15f;
    }

    public boolean debug() {return this.debug;}
    public boolean trackTrainOutput() {return this.trackTrainOutput;}
    public boolean showDistributions() {return this.showDistributions;}

    public int inputSize() {return this.inputSize;}
    public int outputSize() {return this.outputSize;}
    public int[] hiddenLayerSizes() {return this.hiddenLayerSizes;}
    public int totalHiddenLayers() {return this.hiddenLayerSizes.length;}
    public int totalLayers() {return this.hiddenLayerSizes.length+1;}

    public int batchSize() {return this.batchSize;}
    public int epochs() {return this.epochs;}
    public double learnRate() {return this.learnRate;}
}
