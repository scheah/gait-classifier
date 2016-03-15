import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Created by Sebastian on 2/23/2016.
 */
public class DataClassifier {

	protected static final String TAG = "DataClassifier";

	protected Instances m_Data;
	protected Classifier m_Classifier;
	protected String[] m_Classes;

	public DataClassifier(String[] classes) {
		m_Classes = classes;
		m_Classifier = new RandomForest();
		m_Data = initializeTransformedDataFormat();
	}

	public void loadData(File file, String userid) throws Exception {
		readData(file, userid);
	}

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
			Random rand = new Random(1);
			eval.crossValidateModel(m_Classifier, m_Data, folds, rand);
			summary = eval.toSummaryString() + "\n" + eval.toClassDetailsString() + "\n" + eval.toMatrixString();
		} catch (Exception e) {
			System.out.println(e.getMessage());
			summary = "An error has occurred.";
		} finally {
			trainClassifier(m_Data);
		}
		return summary;
	}

	protected void readData(File file, String userid) throws Exception {
		Instances rawData = initializeRawDataFormat();

		if (file == null)
			return;

		BufferedReader reader = new BufferedReader(new FileReader(file));
		String mLine;

		while ((mLine = reader.readLine()) != null) {
			if (mLine.contains(":") || mLine.contains("="))
				continue;
			else if (mLine.contentEquals("")) {
				if (rawData.numInstances() != 0) {
					Instance iDataPoint = transformData(rawData, userid);
					m_Data.add(iDataPoint);
					rawData.clear();
				}
			} else {
				String[] data = mLine.split("\\s+");
				Instance iDataPoint = new DenseInstance(3);
				iDataPoint.setValue(rawData.attribute("x"), Double.parseDouble(data[0]));
				iDataPoint.setValue(rawData.attribute("y"), Double.parseDouble(data[1]));
				iDataPoint.setValue(rawData.attribute("z"), Double.parseDouble(data[2]));
				rawData.add(iDataPoint);
			}
		}

		if (rawData.numInstances() != 0) {
			Instance iDataPoint = transformData(rawData, userid);
			m_Data.add(iDataPoint);
		}

		reader.close();
	}

	protected void trainClassifier(Instances trainData) {
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

	protected String testClassifier(Instances trainData, Instances testData) {
		try {
			Evaluation eTest = new Evaluation(trainData);
			eTest.evaluateModel(m_Classifier, testData);
			return eTest.toSummaryString();
		} catch (Exception e) {
			System.out.println(e.getMessage());
		}
		return null;
	}

	protected Instances initializeRawDataFormat() {
		Attribute attribute1 = new Attribute("x");
		Attribute attribute2 = new Attribute("y");
		Attribute attribute3 = new Attribute("z");

		FastVector fvWekaAttributes = new FastVector(3);
		fvWekaAttributes.addElement(attribute1);
		fvWekaAttributes.addElement(attribute2);
		fvWekaAttributes.addElement(attribute3);

		Instances rawData = new Instances("raw_accel", fvWekaAttributes, 250);
		return rawData;
	}

	protected Instances initializeTransformedDataFormat() {
		return null;
	}

	protected Instance transformData(Instances rawData, String userid) {
		return null;
	}

}
