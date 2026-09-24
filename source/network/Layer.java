package source.network;

import java.util.function.Function;

import source.util.NNMath;

public class Layer {
    int prevLayerSize;
    int size;

    double[][] weights;
    double[] biases;

    double[] activations;
    double[] errors;
    double[][] weightGrads;
    double[] biasGrads;

    Function<double[], double[]> activFun;

    private Layer(int size, int prevLayerSize){
        this.prevLayerSize = prevLayerSize;
        this.size = size;
        this.biases = new double[size];
        this.weights = new double[size][prevLayerSize];
        
        this.activations = new double[size];
        this.errors = new double[size];

        this.weightGrads = new double[size][prevLayerSize];
        this.biasGrads = new double[size];
    }
    
    public Layer(int size, int prevLayerSize, ActivFunction activFun){
        this.prevLayerSize = prevLayerSize;
        this.size = size;
        this.biases = new double[size];
        this.weights = new double[size][prevLayerSize];
        
        this.activations = new double[size];
        this.errors = new double[size];

        this.weightGrads = new double[size][prevLayerSize];
        this.biasGrads = new double[size];

        this.activFun = activFun.function();
    }

    

    public int size() {return this.size;}
    public int prevLayerSize() {return this.prevLayerSize;}

    public double[][] weights() {return this.weights;}
    public double[] bias() {return this.biases;}

    public double[] activations() {return this.activations;}
    public double[] errors() {return this.errors;}
    
    public double[][] weightGrads() {return this.weightGrads;}
    public double[] biasGrads() {return this.biasGrads;}

    /* TODO: not implemented yet */
    public void adjustPrevLayerSize(int newPrevLayerSize) {}

    public void clearGrads() {
        for(int i=0; i<this.size(); i++){
            for(int j=0; j<this.prevLayerSize; j++) this.weightGrads[i][j] = 0.f;
            this.biases[i] = 0.f;
        }
    }

    public void setWeightsBiases(double[][] weights, double[] biases) {
        this.weights = weights;
        this.biases = biases;
    }

    public double[] computeActivations(double[] prevLayerActivs) {
        double[] curr = new double[this.size];
        for(int i=0; i<this.size; i++) {
            curr[i] = NNMath.vDot(this.weights[i], prevLayerActivs);
        }
        this.activations = this.activFun.apply(NNMath.vAdd(this.biases, curr));
        return this.activations;
    }
}
