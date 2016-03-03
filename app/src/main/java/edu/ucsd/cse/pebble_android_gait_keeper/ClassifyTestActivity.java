package edu.ucsd.cse.pebble_android_gait_keeper;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.LinearLayout;
import android.widget.TextView;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;

public class ClassifyTestActivity extends Activity {
    private static final String TAG = "ClassifyTestActivity";

    /* The training data gathered so far. */
    private GaitClassifier m_classifier = new GaitClassifier(new String[]{"brush_teeth", "climb_stairs", "comb_hair",
            "descend_stairs", "drink_glass", "eat_meat", "eat_soup", "getup_bed", "liedown_bed", "pour_water", "sitdown_chair",
            "standup_chair", "use_telephone", "walk"});


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_classify_test);

        BufferedReader reader = null;
        String directory = null;
        String [] list;
        try {
            directory = "HMP_Dataset/Brush_teeth";
            list = getAssets().list(directory);
            for (int i = 0; i < list.length; i++) {
                reader = new BufferedReader(new InputStreamReader(getAssets().open(directory + "/" + list[i])));
                m_classifier.loadGaitData(reader, "brush_teeth");
                reader.close();
            }
            directory = "HMP_Dataset/Climb_stairs";
            list = getAssets().list(directory);
            for (int i = 0; i < list.length; i++) {
                reader = new BufferedReader(new InputStreamReader(getAssets().open(directory + "/" + list[i])));
                m_classifier.loadGaitData(reader, "climb_stairs");
                reader.close();
            }
            directory = "HMP_Dataset/Comb_hair";
            list = getAssets().list(directory);
            for (int i = 0; i < list.length; i++) {
                reader = new BufferedReader(new InputStreamReader(getAssets().open(directory + "/" + list[i])));
                m_classifier.loadGaitData(reader, "comb_hair");
                reader.close();
            }
            directory = "HMP_Dataset/Descend_stairs";
            list = getAssets().list(directory);
            for (int i = 0; i < list.length; i++) {
                reader = new BufferedReader(new InputStreamReader(getAssets().open(directory + "/" + list[i])));
                m_classifier.loadGaitData(reader, "descend_stairs");
                reader.close();
            }
            directory = "HMP_Dataset/Drink_glass";
            list = getAssets().list(directory);
            for (int i = 0; i < list.length; i++) {
                reader = new BufferedReader(new InputStreamReader(getAssets().open(directory + "/" + list[i])));
                m_classifier.loadGaitData(reader, "drink_glass");
                reader.close();
            }
            directory = "HMP_Dataset/Eat_meat";
            list = getAssets().list(directory);
            for (int i = 0; i < list.length; i++) {
                reader = new BufferedReader(new InputStreamReader(getAssets().open(directory + "/" + list[i])));
                m_classifier.loadGaitData(reader, "eat_meat");
                reader.close();
            }
            directory = "HMP_Dataset/Eat_soup";
            list = getAssets().list(directory);
            for (int i = 0; i < list.length; i++) {
                reader = new BufferedReader(new InputStreamReader(getAssets().open(directory + "/" + list[i])));
                m_classifier.loadGaitData(reader, "eat_soup");
                reader.close();
            }
            directory = "HMP_Dataset/Getup_bed";
            list = getAssets().list(directory);
            for (int i = 0; i < list.length; i++) {
                reader = new BufferedReader(new InputStreamReader(getAssets().open(directory + "/" + list[i])));
                m_classifier.loadGaitData(reader, "getup_bed");
                reader.close();
            }
            directory = "HMP_Dataset/Liedown_bed";
            list = getAssets().list(directory);
            for (int i = 0; i < list.length; i++) {
                reader = new BufferedReader(new InputStreamReader(getAssets().open(directory + "/" + list[i])));
                m_classifier.loadGaitData(reader, "liedown_bed");
                reader.close();
            }
            directory = "HMP_Dataset/Pour_water";
            list = getAssets().list(directory);
            for (int i = 0; i < list.length; i++) {
                reader = new BufferedReader(new InputStreamReader(getAssets().open(directory + "/" + list[i])));
                m_classifier.loadGaitData(reader, "pour_water");
                reader.close();
            }
            directory = "HMP_Dataset/Sitdown_chair";
            list = getAssets().list(directory);
            for (int i = 0; i < list.length; i++) {
                reader = new BufferedReader(new InputStreamReader(getAssets().open(directory + "/" + list[i])));
                m_classifier.loadGaitData(reader, "sitdown_chair");
                reader.close();
            }
            directory = "HMP_Dataset/Standup_chair";
            list = getAssets().list(directory);
            for (int i = 0; i < list.length; i++) {
                reader = new BufferedReader(new InputStreamReader(getAssets().open(directory + "/" + list[i])));
                m_classifier.loadGaitData(reader, "standup_chair");
                reader.close();
            }
            directory = "HMP_Dataset/Use_telephone";
            list = getAssets().list(directory);
            for (int i = 0; i < list.length; i++) {
                reader = new BufferedReader(new InputStreamReader(getAssets().open(directory + "/" + list[i])));
                m_classifier.loadGaitData(reader, "use_telephone");
                reader.close();
            }
            directory = "HMP_Dataset/Walk";
            list = getAssets().list(directory);
            for (int i = 0; i < list.length; i++) {
                reader = new BufferedReader(new InputStreamReader(getAssets().open(directory + "/" + list[i])));
                m_classifier.loadGaitData(reader, "walk");
                reader.close();
            }
        }
        catch (Exception e) {
            e.printStackTrace();
            Log.d(TAG, e.getMessage());
        }

        m_classifier.train();
        LinearLayout layout = (LinearLayout) findViewById(R.id.classifyTestActivityLinearLayout);

        TextView summary = new TextView(this);
        summary.setText("Summary of Classifier: \n" + m_classifier.dataSummary());
        summary.setLayoutParams(new LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT));
        layout.addView(summary);

        TextView testSummary = new TextView(this);
        testSummary.setText("Internal Testing summary: \n" + m_classifier.internalTestSummary(5));
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
