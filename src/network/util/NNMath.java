package src.network.util;

import java.util.Arrays;

public class NNMath {
    // Performs dot product on 2 arrays of equal size and returns resulting scalar
    public static double vDot(double[] a, double[] b) {
        assert a.length == b.length : "Incompatible vector sizes for dot product";
        double result = 0.0f;
        for(int i=0; i<a.length; i++) result += a[i]*b[i];
        return result;
    }

    // Performs vector component addition for 2 arrays of equal size and return resulting vector
    public static double[] vAdd(double[] a, double[] b) {
        assert a.length == b.length : "Incompatible vector sizes for addition";
        double[] result = new double[a.length];
        for(int i=0; i<a.length; i++) result[i] = a[i]+b[i];
        return result;
    }

    public static double[] sigmoid(double[] z) {
        double[] a = new double[z.length];
        for(int i=0; i<z.length; i++) {
            a[i] = 1/(1+(double)Math.exp(-z[i]));
        }
        return a;
    }

    public static double[] reLU(double[] z) {
        double[] a = new double[z.length];
        for(int i=0; i<z.length; i++) {
            a[i] = Math.max(0, z[i]);
        }
        return a;
    }

    public static double[] softmax(double[] z) {
        double[] d = new double[z.length];
        double wtdsum = Arrays.stream(z).map(x -> Math.exp(x)).sum();
        for(int i=0; i<z.length; i++){
            d[i] = Math.exp(z[i])/wtdsum;
        } 
        return d;
    }
}