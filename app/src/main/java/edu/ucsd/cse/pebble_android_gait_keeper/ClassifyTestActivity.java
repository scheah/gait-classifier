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

public class ClassifyTestActivity extends Activity {
    private static final String TAG = "ClassifyTestActivity";

    /* The training data gathered so far. */
    private GaitClassifier m_classifier = new GaitClassifier();


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_classify_test);

        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new InputStreamReader(getAssets().open("data_scott_1.log")));
            m_classifier.train(reader);
        }
        catch (Exception e) {
            e.printStackTrace();
            Log.d(TAG, e.getMessage());
        }
        finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    e.printStackTrace();
                    Log.d(TAG, e.getMessage());
                }
            }
        }

        LinearLayout layout = (LinearLayout) findViewById(R.id.classifyTestActivityLinearLayout);

        TextView summary = new TextView(this);
        summary.setText("Summary of Classifier: \n" + m_classifier.dataSummary());
        summary.setLayoutParams(new LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT));
        layout.addView(summary);

        TextView testSummary = new TextView(this);
        testSummary.setText("Internal Testing summary: \n" + m_classifier.internalTestSummary(20));
        testSummary.setLayoutParams(new LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT));
        layout.addView(testSummary);

        TextView dataDump = new TextView(this);
        dataDump.setText(m_classifier.dataDump());
        dataDump.setLayoutParams(new LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT));
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
}
