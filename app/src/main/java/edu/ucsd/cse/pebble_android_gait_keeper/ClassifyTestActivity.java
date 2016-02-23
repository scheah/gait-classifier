package edu.ucsd.cse.pebble_android_gait_keeper;

import android.app.Activity;
import android.os.Bundle;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.LinearLayout;
import android.widget.TextView;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.FastVector;
import weka.classifiers.Classifier;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;


public class ClassifyTestActivity extends Activity {
    private static final String TAG = "ClassifyTestActivity";

    /* The training data gathered so far. */
    private Instances m_Data = null;
    private Instances m_TrainingData = null;
    private Instances m_TestData = null;
    private Classifier m_Classifier = new RandomForest();


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_classify_test);

        this.readData();
        this.trainClassifier();

        LinearLayout layout = (LinearLayout) findViewById(R.id.classifyTestActivityLinearLayout);
        TextView readDataResults = new TextView(this);
        readDataResults.setText("All data instances read: " + Integer.toString(m_Data.numInstances()) +
                "\nTraining instances: " + Integer.toString(m_TrainingData.numInstances()) +
                "\nTesting instances: " + Integer.toString(m_TestData.numInstances()));
        readDataResults.setLayoutParams(new LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT));
        layout.addView(readDataResults);
        TextView testResults = new TextView(this);
        testResults.setText("Accuracy: TBD");
        testResults.setLayoutParams(new LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT));
        layout.addView(testResults);

        TextView dataDump = new TextView(this);
        dataDump.setText(m_Data.toString());
        dataDump.setLayoutParams(new LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT));
        dataDump.setMovementMethod(new ScrollingMovementMethod());
        layout.addView(dataDump);
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_classify_test, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }

    public void readData() {

        double percentageSplit = 20; // Define % for data split for training/test set (80/20)

        Instances rawData = initializeRawDataFormat();
        Instances transformedData = initializeTransformedDataFormat();
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new InputStreamReader(getAssets().open("data_scott_1.log")));
            // do reading, usually loop until end of file reading
            String mLine;
            while ((mLine = reader.readLine()) != null) {
                //process line
                Instance iDataPoint = null;
                if (mLine.contains(":"))
                    continue; // skip line
                else if (mLine.contentEquals("")){
                    // End of sample window. Extract transformed features now
                    iDataPoint = transformData(rawData, transformedData);
                    transformedData.add(iDataPoint);
                    rawData.clear(); // reset to read in new window
                }
                else {
                    String[] data = mLine.split("\\s+");
                    // build and add instance
                    iDataPoint = new DenseInstance(4);
                    iDataPoint.setValue(rawData.attribute("x"), Double.parseDouble(data[0]));
                    iDataPoint.setValue(rawData.attribute("y"), Double.parseDouble(data[1]));
                    iDataPoint.setValue(rawData.attribute("z"), Double.parseDouble(data[2]));
                    rawData.add(iDataPoint);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
            Log.d(TAG, e.getMessage());
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    e.printStackTrace();
                    Log.d(TAG, e.getMessage());
                }
            }
        }
        m_Data = transformedData;
        m_TrainingData = getTrainingSet(m_Data, percentageSplit);
        m_TestData = getTestSet(m_Data, percentageSplit);
    }

    public Instance transformData(Instances rawData, Instances transformedData) {
        String [] axis = new String[] {"x", "y", "z"};
        double [] mean = new double[axis.length];
        Instance windowDataPoint = new DenseInstance(44); // 43 features + 1 for class
        // Average and Standard Deviation
        for (int i = 0; i < axis.length; i++) {
            mean[i] = rawData.meanOrMode(i);
            double sd = Math.sqrt(rawData.variance(i));
            windowDataPoint.setValue(transformedData.attribute("Average " + axis[i]), mean[i]);
            windowDataPoint.setValue(transformedData.attribute("Standard Deviation " + axis[i]), sd);
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
            windowDataPoint.setValue(transformedData.attribute("Average Absolute Difference " + axis[i]), avgAbsDiff[i]);
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
            windowDataPoint.setValue(transformedData.attribute("Time between peaks " + axis[i]), sumTimeBetwnPeaks/numPeaks); // TBD
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
                if (currInst.value(2) <= rangesX[i]) {
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
                    Log.d(TAG, "[transformData(Instances rawData)] Possible Error");
            }
            windowDataPoint.setValue(transformedData.attribute("Binned Distribution " + axis[dim] + " " + Integer.toString(binNum)), dimBins[binNum]);
        }
        // Average Resultant Acceleration
        double avgResultAccel = 0;
        for (int instIdx = 0; instIdx < rawData.numInstances(); instIdx++) {
            Instance currInst = rawData.instance(instIdx);
            avgResultAccel += Math.sqrt(Math.pow(currInst.value(0), 2) + Math.pow(currInst.value(1), 2) + Math.pow(currInst.value(2), 2));
        }
        avgResultAccel /= rawData.numInstances();
        windowDataPoint.setValue(transformedData.attribute("Average Resultant Acceleration"), avgResultAccel);
        // Class attribute
        windowDataPoint.setValue(transformedData.attribute("userid"), "user1");

        return windowDataPoint;
    }

    public Instances initializeRawDataFormat() {
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
        Instances rawData = new Instances("raw_gait" /*relation name*/, fvWekaAttributes /*attribute vector*/, 250 /*initial capacity*/);
        return rawData;
    }

    public Instances initializeTransformedDataFormat() {
        int numAttributes = 44;
        String [] axis = new String[] {"x", "y", "z"};
        String[] classes = new String[]{"user1", "user2"}; // TBD: add more depending on gathered data

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
        FastVector fvUserVal = new FastVector(2);
        for (int i = 0; i < classes.length; i++) {
            fvUserVal.addElement(classes[i]);
        }
        Attribute classAttribute = new Attribute("userid", fvUserVal);
        fvTransformedWekaAttributes.addElement(classAttribute);
        Instances ret = new Instances("gait" /*relation name*/, fvTransformedWekaAttributes /*attribute vector*/, 25 /*initial capacity*/);
        ret.setClassIndex(ret.numAttributes() - 1);

        return ret;
    }

    public Instances getTrainingSet(Instances data, double percentSplit) {
        Instances trainingSet = null;
        try {
            // Removes 20%
            RemovePercentage dataSplitter = new RemovePercentage();
            dataSplitter.setPercentage(percentSplit);
            dataSplitter.setInputFormat(data);
            trainingSet = Filter.useFilter(data, dataSplitter);
        } catch (Exception e) {
            e.printStackTrace();
            Log.d(TAG, e.getMessage());
        }
        return trainingSet;
    }

    public Instances getTestSet(Instances data, double percentSplit) {
        Instances testSet = null;
        try {
            // Do the opposite to get the remaining data
            RemovePercentage dataSplitter = new RemovePercentage();
            dataSplitter.setInvertSelection(true);
            dataSplitter.setPercentage(percentSplit);
            dataSplitter.setInputFormat(data);
            testSet = Filter.useFilter(data, dataSplitter);
        } catch (Exception e) {
            e.printStackTrace();
            Log.d(TAG, e.getMessage());
        }
        return testSet;
    }

    public void trainClassifier() {
        // Call readTrainingData first!
        // Check whether training data has been built.
        if (m_Data.numInstances() == 0) {
            Log.d(TAG, "No training data available");
            return;
        }
        try {
            m_Classifier.buildClassifier(m_Data);
        } catch (Exception e) {
            e.printStackTrace();
            Log.d(TAG, e.getMessage());
        }
    }
}
