import java.io.BufferedReader;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 * Created by Sebastian on 2/23/2016.
 */
public class GaitClassifier {
    private static final String TAG = "GaitClassifier";
    private Instances m_Data;
    private Classifier m_Classifier;
    private String [] m_Classes;

    public GaitClassifier(String [] classes) {
        m_Classes = classes;
        m_Data = initializeTransformedDataFormat2();
        m_Classifier = new RandomForest();
    }

    public void loadGaitData(BufferedReader reader, String userid) throws Exception {
        readData(reader, userid);
    }

    // Clear m_Data, and resets classifier
    public void clearData() {
        m_Data.clear();
        m_Classifier = new RandomForest();
    }

    public void train() {
        trainClassifier(m_Data);
    }

    public double classifyInstance(Instance instance) throws Exception {
        return m_Classifier.classifyInstance(instance);
    }

    public String dataDump() {
        return m_Data.toString();
    }

    public String dataSummary() {
        return m_Data.toSummaryString();
    }

    public String internalTestSummary(int folds) {
        String summary;
        if (m_Data.numInstances() == 0) {
            return "No Data Available!";
        }
        try {
            Evaluation eval = new Evaluation(m_Data);
            Random rand = new Random(1);  // using seed = 1
            eval.crossValidateModel(m_Classifier, m_Data, folds, rand);
            summary = eval.toSummaryString() + "\n" + eval.toClassDetailsString() + "\n" + eval.toMatrixString();
        } catch (Exception e) {
            System.out.println(e.getMessage());
            summary = "An error has occurred.";
        } finally {
            trainClassifier(m_Data); // Reset classifier back to full training set
        }
        return summary;
    }

    // Read from file system
    private void readData(BufferedReader reader, String userid) throws Exception {
        Instances rawData = initializeRawDataFormat();
        if (reader == null)
            return;
        String mLine;
        while ((mLine = reader.readLine()) != null) {
            //process line
            Instance iDataPoint = null;
            if (mLine.contains(":") || mLine.contains("="))
                continue; // skip line
            else if (mLine.contentEquals("")) {
                // End of sample window. Extract transformed features now
                if (rawData.numInstances() != 0) { // skip any blank line if we haven't read in any data yet
                    iDataPoint = transformData2(rawData, userid);
                    m_Data.add(iDataPoint);
                    rawData.clear(); // reset to read in new window
                }
            } else {
                String[] data = mLine.split("\\s+");
                // build and add instance
                iDataPoint = new DenseInstance(4);
                iDataPoint.setValue(rawData.attribute("x"), Double.parseDouble(data[0]));
                iDataPoint.setValue(rawData.attribute("y"), Double.parseDouble(data[1]));
                iDataPoint.setValue(rawData.attribute("z"), Double.parseDouble(data[2]));
                rawData.add(iDataPoint);
            }
        }
        if (rawData.numInstances() != 0) { // handle potential window near end of file
            Instance iDataPoint = transformData2(rawData, userid);
            m_Data.add(iDataPoint);
        }
    }

    private void trainClassifier(Instances trainData) {
        // Check whether training data has been built.
        if (trainData.numInstances() == 0) {
            System.out.println("No training data available");
            return;
        }
        try {
            m_Classifier.buildClassifier(trainData);
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }

    private String testClassifier(Instances trainData, Instances testData) {
        try {
            Evaluation eTest = new Evaluation(trainData);
            eTest.evaluateModel(m_Classifier, testData);
            return eTest.toSummaryString();
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
        return null;
    }

    private Instances initializeTransformedDataFormat() {
        int numAttributes = 44;
        String [] axis = new String[] {"x", "y", "z"};
        String [] classes = m_Classes;
        // Transformed Data
        FastVector fvTransformedWekaAttributes = new FastVector(numAttributes);
        // Numeric attributes
        // Average
        for (int i = 0; i < axis.length; i++) {
            Attribute attribute = new Attribute("Average " + axis[i]);
            fvTransformedWekaAttributes.addElement(attribute);
        }
        // Standard Deviation
        for (int i = 0; i < axis.length; i++) {
            Attribute attribute = new Attribute("Standard Deviation " + axis[i]);
            fvTransformedWekaAttributes.addElement(attribute);
        }
        // Average Absolute Difference
        for (int i = 0; i < axis.length; i++) {
            Attribute attribute = new Attribute("Average Absolute Difference " + axis[i]);
            fvTransformedWekaAttributes.addElement(attribute);
        }
        // Time between peaks
        for (int i = 0; i < axis.length; i++) {
            Attribute attribute = new Attribute("Time between peaks " + axis[i]);
            fvTransformedWekaAttributes.addElement(attribute);
        }
        // Binned Distribution
        for (int i = 0; i < 10 * axis.length; i++) {
            Attribute attribute = new Attribute("Binned Distribution " + axis[i/10] + " " + Integer.toString(i % 10));
            fvTransformedWekaAttributes.addElement(attribute);
        }
        // Average Resultant Acceleration
        Attribute attribute = new Attribute("Average Resultant Acceleration");
        fvTransformedWekaAttributes.addElement(attribute);
        // Declare class attribute
        FastVector fvUserVal = new FastVector(classes.length);
        for (int i = 0; i < classes.length; i++) {
            fvUserVal.addElement(classes[i]);
        }
        Attribute classAttribute = new Attribute("userid", fvUserVal);
        fvTransformedWekaAttributes.addElement(classAttribute);
        Instances transformedData = new Instances("gait" /*relation name*/, fvTransformedWekaAttributes /*attribute vector*/, 25 /*initial capacity*/);
        transformedData.setClassIndex(transformedData.numAttributes() - 1);
        return transformedData;
    }

    private Instances initializeTransformedDataFormat2() {
        int numAttributes = 19;
        String [] axis = new String[] {"x", "y", "z"};
        String [] classes = m_Classes;
        // Transformed Data
        FastVector fvTransformedWekaAttributes = new FastVector(numAttributes);
        // Numeric attributes
        // Mean
        for (int i = 0; i < axis.length; i++) {
            Attribute attribute = new Attribute("Mean " + axis[i]);
            fvTransformedWekaAttributes.addElement(attribute);
        }
        // Variance
        for (int i = 0; i < axis.length; i++) {
            Attribute attribute = new Attribute("Variance " + axis[i]);
            fvTransformedWekaAttributes.addElement(attribute);
        }
        // Standard Deviation
        for (int i = 0; i < axis.length; i++) {
            Attribute attribute = new Attribute("Standard Deviation " + axis[i]);
            fvTransformedWekaAttributes.addElement(attribute);
        }
        // Minimum
        for (int i = 0; i < axis.length; i++) {
            Attribute attribute = new Attribute("Minimum " + axis[i]);
            fvTransformedWekaAttributes.addElement(attribute);
        }
        // Average Absolute Sample Difference
        for (int i = 0; i < axis.length; i++) {
            Attribute attribute = new Attribute("Average Absolute Sample Difference " + axis[i]);
            fvTransformedWekaAttributes.addElement(attribute);
        }
        // Correlation
        for (int i = 0; i < axis.length-1; i++) {
            for (int j = i + 1; j < axis.length; j++) {
                Attribute attribute = new Attribute("Correlation " + axis[i] + axis[j]);
                fvTransformedWekaAttributes.addElement(attribute);
            }
        }
        // Declare class attribute
        FastVector fvUserVal = new FastVector(classes.length);
        for (int i = 0; i < classes.length; i++) {
            fvUserVal.addElement(classes[i]);
        }
        Attribute classAttribute = new Attribute("userid", fvUserVal);
        fvTransformedWekaAttributes.addElement(classAttribute);
        Instances transformedData = new Instances("FeaturesV2" /*relation name*/, fvTransformedWekaAttributes /*attribute vector*/, 25 /*initial capacity*/);
        transformedData.setClassIndex(transformedData.numAttributes() - 1);
        return transformedData;
    }

    private Instances initializeRawDataFormat() {
        // Raw Data Points to extract features from
        Attribute attribute1 = new Attribute("x");
        Attribute attribute2 = new Attribute("y");
        Attribute attribute3 = new Attribute("z");
        // Declare the feature vector
        FastVector fvWekaAttributes = new FastVector(3);
        fvWekaAttributes.addElement(attribute1);
        fvWekaAttributes.addElement(attribute2);
        fvWekaAttributes.addElement(attribute3);
        // Create an empty raw data set
        Instances rawData = new Instances("raw_accel" /*relation name*/, fvWekaAttributes /*attribute vector*/, 250 /*initial capacity*/);
        return rawData;
    }

    private Instance transformData(Instances rawData, String userid) {
        String [] axis = new String[] {"x", "y", "z"};
        double [] mean = new double[axis.length];
        Instance windowDataPoint = new DenseInstance(44); // 43 features + 1 for class
        // Average and Standard Deviation
        for (int i = 0; i < axis.length; i++) {
            mean[i] = rawData.meanOrMode(i);
            double sd = Math.sqrt(rawData.variance(i));
            windowDataPoint.setValue(m_Data.attribute("Average " + axis[i]), mean[i]);
            windowDataPoint.setValue(m_Data.attribute("Standard Deviation " + axis[i]), sd);
        }
        // Average Absolute Difference
        double [] avgAbsDiff = new double[axis.length];
        for (int instIdx = 0; instIdx < rawData.numInstances(); instIdx++) {
            Instance currInst = rawData.instance(instIdx);
            for (int i = 0; i < axis.length; i++) {
                avgAbsDiff[i] += Math.abs( currInst.value(rawData.attribute(axis[i])) - mean[i] );
            }
        }
        for (int i = 0; i < axis.length; i++) {
            avgAbsDiff[i] /= rawData.numInstances();
            windowDataPoint.setValue(m_Data.attribute("Average Absolute Difference " + axis[i]), avgAbsDiff[i]);
        }
        // Time Between Peaks
        boolean [][] peaks = new boolean[rawData.numAttributes()][rawData.numInstances()];
        int dt = 1; // time delay of detection
        int m = 1; // peak threshold detection
        for (int instIdx = 0; instIdx < rawData.numInstances(); instIdx++) {
            int lowerBound = instIdx - dt;
            int upperBound = instIdx + dt;
            if (lowerBound < 0)
                lowerBound = 0;
            if (upperBound >= rawData.numInstances())
                upperBound = rawData.numInstances() - 1;
            Instance lowerInst = rawData.instance(lowerBound);
            Instance currInst = rawData.instance(instIdx);
            Instance upperInst = rawData.instance(upperBound);
            for (int i = 0; i < axis.length; i++) {
                if (currInst.value(i) - lowerInst.value(i) > m && currInst.value(i) - upperInst.value(i) > m) {
                    peaks[i][instIdx] = true;
                }
            }
        }
        for (int i = 0; i < axis.length; i++) {
            double sumTimeBetwnPeaks = 0;
            double timeBetwnPeaks = 0;
            double numPeaks = 0;
            for (int j = 0; j < peaks[i].length; j++) {
                if (peaks[i][j] == true) {
                    if (numPeaks != 0) {
                        sumTimeBetwnPeaks += timeBetwnPeaks;
                    }
                    timeBetwnPeaks = 0;
                    numPeaks++;
                }
                else {
                    timeBetwnPeaks++;
                }
            }
            windowDataPoint.setValue(m_Data.attribute("Time between peaks " + axis[i]), sumTimeBetwnPeaks/numPeaks); // TBD
        }
        // Binned Distribution
        double [] binsX = new double[10];
        double [] binsY = new double[10];
        double [] binsZ = new double[10];
        double [] rangesX = new double[10];
        double [] rangesY = new double[10];
        double [] rangesZ = new double[10];
        double minimumX = rawData.attributeStats(0).numericStats.min;
        double maximumX = rawData.attributeStats(0).numericStats.max;
        double minimumY = rawData.attributeStats(1).numericStats.min;
        double maximumY = rawData.attributeStats(1).numericStats.max;
        double minimumZ = rawData.attributeStats(2).numericStats.min;
        double maximumZ = rawData.attributeStats(2).numericStats.max;
        double stepX = (maximumX - minimumX) / 10;
        double stepY = (maximumY - minimumY) / 10;
        double stepZ = (maximumZ - minimumZ) / 10;
        for (int i = 0; i < 10; i++) {
            rangesX[i] = minimumX + stepX * (i+1);
            rangesY[i] = minimumY + stepY * (i+1);
            rangesZ[i] = minimumZ + stepZ * (i+1);
        }
        for (int instIdx = 0; instIdx < rawData.numInstances(); instIdx++) {
            Instance currInst = rawData.instance(instIdx);
            for (int i = 0; i < 10; i++) {
                if (currInst.value(0) <= rangesX[i]) {
                    binsX[i] += 1;
                    break;
                }
            }
            for (int i = 0; i < 10; i++) {
                if (currInst.value(1) <= rangesY[i]) {
                    binsY[i] += 1;
                    break;
                }
            }
            for (int i = 0; i < 10; i++) {
                if (currInst.value(2) <= rangesZ[i]) {
                    binsZ[i] += 1;
                    break;
                }
            }
        }
        for (int i = 0; i < 10; i++) {
            binsX[i] /= rawData.numInstances();
            binsY[i] /= rawData.numInstances();
            binsZ[i] /= rawData.numInstances();
        }
        for (int i = 0; i < 10 * axis.length; i++) {
            int dim = i/10;
            int binNum = i % 10;
            double [] dimBins = null;
            switch(dim) {
                case 0:
                    dimBins = binsX;
                    break;
                case 1:
                    dimBins = binsY;
                    break;
                case 2:
                    dimBins = binsZ;
                    break;
                default:
                    System.out.println("[transformData(Instances rawData)] Possible Error");
            }
            windowDataPoint.setValue(m_Data.attribute("Binned Distribution " + axis[dim] + " " + Integer.toString(binNum)), dimBins[binNum]);
        }
        // Average Resultant Acceleration
        double avgResultAccel = 0;
        for (int instIdx = 0; instIdx < rawData.numInstances(); instIdx++) {
            Instance currInst = rawData.instance(instIdx);
            avgResultAccel += Math.sqrt(Math.pow(currInst.value(0), 2) + Math.pow(currInst.value(1), 2) + Math.pow(currInst.value(2), 2));
        }
        avgResultAccel /= rawData.numInstances();
        windowDataPoint.setValue(m_Data.attribute("Average Resultant Acceleration"), avgResultAccel);
        // Class attribute
        windowDataPoint.setValue(m_Data.attribute("userid"), userid);

        return windowDataPoint;
    }

    private Instance transformData2(Instances rawData, String userid) {
        String [] axis = new String[] {"x", "y", "z"};
        Instance windowDataPoint = new DenseInstance(19);
        // Mean, Variance, Standard Deviation, Minimum
        for (int i = 0; i < axis.length; i++) {
            double mean = rawData.meanOrMode(i);
            double var = rawData.variance(i);
            double sd = Math.sqrt(var);
            double minimum = rawData.attributeStats(i).numericStats.min;
            windowDataPoint.setValue(m_Data.attribute("Mean " + axis[i]), mean);
            windowDataPoint.setValue(m_Data.attribute("Variance " + axis[i]), var);
            windowDataPoint.setValue(m_Data.attribute("Standard Deviation " + axis[i]), sd);
            windowDataPoint.setValue(m_Data.attribute("Minimum " + axis[i]), minimum);
        }
        // Average Absolute Sample Difference
        double [] sumDeltas = new double[] {0,0,0};
        Instance prevInst = rawData.instance(0);
        for (int instIdx = 1; instIdx < rawData.numInstances(); instIdx++) {
            Instance currInst = rawData.instance(instIdx);
            for (int i = 0; i < axis.length; i++) {
                sumDeltas[i] += currInst.value(rawData.attribute(axis[i])) - prevInst.value(rawData.attribute(axis[i]));
            }
            prevInst = currInst;
        }
        for (int i = 0; i < axis.length; i++) {
            windowDataPoint.setValue(m_Data.attribute("Average Absolute Sample Difference " + axis[i]), sumDeltas[i]/rawData.numInstances());
        }
        for (int i = 0; i < axis.length-1; i++) {
            for (int j = i + 1; j < axis.length; j++) {
                double [] arrI = rawData.attributeToDoubleArray(i);
                double [] arrJ = rawData.attributeToDoubleArray(j);
                windowDataPoint.setValue(m_Data.attribute("Correlation " + axis[i] + axis[j]), Utils.correlation(arrI, arrJ, arrI.length));
            }
        }
        // Class attribute
        windowDataPoint.setValue(m_Data.attribute("userid"), userid);

        return windowDataPoint;
    }

}
