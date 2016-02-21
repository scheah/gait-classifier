package edu.ucsd.cse.pebble_android_gait_keeper;

import android.app.Activity;
import android.os.Bundle;
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
        int numAttributes = 4; // Attribute format: (x, y, z, UserID)
        double percentageSplit = 20; // Define % for data split for training/test set (80/20)

        // Declare three numeric attributes
        Attribute attribute1 = new Attribute("x");
        Attribute attribute2 = new Attribute("y");
        Attribute attribute3 = new Attribute("z");
        // Declare the class attribute along with its values
        FastVector fvClassVal = new FastVector(2);
        String[] classes = new String[]{"user1", "user2"}; // TBD: add more depending on gathered data
        for (int i = 0; i < classes.length; i++) {
            fvClassVal.addElement(classes[i]);
        }
        Attribute classAttribute = new Attribute("theClass", fvClassVal);
        // Declare the feature vector
        FastVector fvWekaAttributes = new FastVector(numAttributes);
        fvWekaAttributes.addElement(attribute1);
        fvWekaAttributes.addElement(attribute2);
        fvWekaAttributes.addElement(attribute3);
        fvWekaAttributes.addElement(classAttribute);
        // Create an empty training set
        m_Data = new Instances("gait" /*relation name*/, fvWekaAttributes /*attribute vector*/, 25*250 /*initial capacity*/);
        // Set class index
        m_Data.setClassIndex(numAttributes-1); // The last index of the attribute vector should contain the class attribute
        // Create instances
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new InputStreamReader(getAssets().open("data_scott_1.log")));
            // do reading, usually loop until end of file reading
            String mLine;
            while ((mLine = reader.readLine()) != null) {
                //process line
                if (mLine.contains(":") || mLine.contentEquals(""))
                    continue; // skip line
                String[] data = mLine.split("\\s+");
                // build and add instance
                Instance iDataPoint = new Instance(4);
                iDataPoint.setValue((Attribute)fvWekaAttributes.elementAt(0), Double.parseDouble(data[0]));
                iDataPoint.setValue((Attribute)fvWekaAttributes.elementAt(1), Double.parseDouble(data[1]));
                iDataPoint.setValue((Attribute)fvWekaAttributes.elementAt(2), Double.parseDouble(data[2]));
                iDataPoint.setValue((Attribute)fvWekaAttributes.elementAt(3), "user1");
                m_Data.add(iDataPoint);
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
        try {
            // Removes 20%
            RemovePercentage dataSplitter = new RemovePercentage();
            dataSplitter.setPercentage(percentageSplit);
            dataSplitter.setInputFormat(m_Data);
            m_TrainingData = Filter.useFilter(m_Data, dataSplitter);
            // Do the opposite to get the remaining data
            dataSplitter = new RemovePercentage();
            dataSplitter.setInvertSelection(true);
            dataSplitter.setPercentage(percentageSplit);
            dataSplitter.setInputFormat(m_Data);
            m_TestData = Filter.useFilter(m_Data, dataSplitter);
        } catch (Exception e) {
            e.printStackTrace();
            Log.d(TAG, e.getMessage());
        }


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
