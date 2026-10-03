package src.network.tracking;

import java.util.ArrayList;
import java.util.List;


public class NetworkTracking {
    int[] totalActualCorrect;
    int[] totalExpectedCorrect;
    int[] epochActualCorrect;
    int[] epochExpectedCorrect;

    List<Misclassification> misclass;

    public NetworkTracking(int outputSize) {
        this.totalActualCorrect = new int[outputSize];
        this.totalExpectedCorrect = new int[outputSize];
        this.epochActualCorrect = new int[outputSize];
        this.epochExpectedCorrect = new int[outputSize];
        this.misclass = new ArrayList<>();
    }

    public void classifyMiss(int expectedIdx, int actualIdx, double[] activations, int caseIdx) {
        this.totalExpectedCorrect[expectedIdx] += 1;
        this.epochExpectedCorrect[expectedIdx] += 1;
        this.misclass.add(new Misclassification(caseIdx, activations.clone()));
    }

    public void classifyCorrect(int expectedIdx, int actualIdx) {
        this.totalExpectedCorrect[expectedIdx] += 1;
        this.epochExpectedCorrect[expectedIdx] += 1;
        this.totalActualCorrect[actualIdx] += 1;
        this.epochActualCorrect[actualIdx] += 1;
    }

    public void clearAllCounts() {
        this.totalActualCorrect = new int[this.totalExpectedCorrect.length];
        this.totalExpectedCorrect = new int[this.totalActualCorrect.length];
        this.clearEpochCounts();
    }

    public void clearEpochCounts() {
        this.epochActualCorrect = new int[this.epochExpectedCorrect.length];
        this.epochExpectedCorrect = new int[this.epochActualCorrect.length];
    }

    public int[] totalActualCorrect() {return this.totalActualCorrect;}

    public int[] totalExpectedCorrect() {return this.totalExpectedCorrect;}

    public int[] epochActualCorrect() {return this.epochActualCorrect;}

    public int[] epochExpectedCorrect() {return this.epochExpectedCorrect;}
}
