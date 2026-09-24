package source.network;

import java.util.function.Function;
import source.util.NNMath;

public enum ActivFunction {
    SIGMOID((z) -> NNMath.sigmoid(z)),
    RELU((z) -> NNMath.reLU(z)),
    SOFTMAX((z) -> NNMath.softmax(z));

    private final Function<double[], double[]> function;

    ActivFunction(Function<double[], double[]> function) {
        this.function = function;
    }

    public Function<double[], double[]> function() {
        return this.function;
    }
}
