package src.network.persist;

import src.network.components.ActivFunction;
import src.network.components.Layer;
import src.network.config.NetworkConfig;

public class NetworkSnapshot {
    String name;
    NetworkConfig config;
    LayerWeightsBiases[] weightsBiases;

    public NetworkSnapshot(String name, NetworkConfig config, Layer[] hiddenLayers, Layer outLayer) {
        this.name = name;
        this.config = config;
        this.weightsBiases = new LayerWeightsBiases[config.totalLayers()];

        for(int l=0; l<hiddenLayers.length; l++) 
            this.weightsBiases[l] = new LayerWeightsBiases(
                                                hiddenLayers[l].weights(),
                                                hiddenLayers[l].bias());

        this.weightsBiases[config.totalLayers()-1] = new LayerWeightsBiases(
                                                                outLayer.weights(),
                                                                outLayer.bias());
    }

    NetworkSnapshot() {}

    public NetworkConfig copyConfig() {return new NetworkConfig(this.config);}

    public LayersSnapshot layersSnapshot() {
        Layer[] hidden = new Layer[this.config.totalHiddenLayers()];

        int prevLayerSize = this.config.inputSize();
        for(int i=0; i<this.config.totalHiddenLayers(); i++) {
            hidden[i] = new Layer(this.config.hiddenLayerSizes()[i], prevLayerSize, ActivFunction.RELU);
            hidden[i].setWeightsBiases(this.weightsBiases[i].weights, this.weightsBiases[i].biases);
            prevLayerSize = hidden[i].size();
        }

        Layer out = new Layer(this.config.outputSize(), prevLayerSize, ActivFunction.SIGMOID);
        out.setWeightsBiases(
                this.weightsBiases[this.weightsBiases.length-1].weights, 
                this.weightsBiases[this.weightsBiases.length-1].biases);

        return new LayersSnapshot(hidden, out);
    }

    /* TODO: Not implemented */
    public void saveSnapshot(String filename) {}

    /* TODO: Not implemented */
    public static NetworkSnapshot loadSnapshot(String filename) {
        return new NetworkSnapshot();
    }

    public NetworkConfig config() {return this.config;}

    public String name() {return this.name;}

    private record LayerWeightsBiases(double[][] weights, double[] biases) {}
}

