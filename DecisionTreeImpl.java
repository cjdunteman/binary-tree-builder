import java.util.List;

import java.util.ArrayList;

/**
 * Fill in the implementation details of the class DecisionTree using this file.
 * Any methods or secondary classes that you want are fine but we will only
 * interact with those methods in the DecisionTree framework.
 */
public class DecisionTreeImpl {

	static final int ATTRIBUTE = 99;
	static final int THRESHOLD = 99;
	static final int LABEL = 99;
	static final int ZERO = 0;
	static final int EIGHT = 8;
	static final int MALIGNANT = 1;
	static final int BENIGN = 0;

	public DecTreeNode root;
	public List<List<Integer>> trainData;
	public int maxPerLeaf;
	public int maxDepth;
	public int numAttr;

	// Build a decision tree given a training set
	DecisionTreeImpl(List<List<Integer>> trainDataSet, int mPerLeaf, int mDepth) {
		this.trainData = trainDataSet;
		this.maxPerLeaf = mPerLeaf;
		this.maxDepth = mDepth;
		List<Attribute> attributeList = new ArrayList<>();

		if (this.trainData.size() > 0)
			this.numAttr = trainDataSet.get(0).size() - 1;

		if (this.numAttr > 0) {
			for (int i = ZERO; i <= EIGHT; i++) {
				for (int j = 0; j <= this.numAttr; j++) {
					attributeList.add(new Attribute(i, j));
				}
			}
		}

		this.root = buildTree(trainDataSet, trainDataSet, attributeList, 0);
	}

	private DecTreeNode buildTree(List<List<Integer>> dataExample, List<List<Integer>> parentEx,
			List<Attribute> attributeList, int depth) {
		if (dataExample.size() < 1) {
			int label = Helpers.value(parentEx);
			return new DecTreeNode(label, ATTRIBUTE, THRESHOLD);

		} else if (Helpers.checkLabels(dataExample)) {
			int label = Helpers.getLabel(dataExample.get(0));
			return new DecTreeNode(label, ATTRIBUTE, THRESHOLD);

		} else if ((depth == this.maxDepth) || (dataExample.size() <= this.maxPerLeaf) || (attributeList.size() < 1)) {
			int label = Helpers.value(dataExample);
			return new DecTreeNode(label, ATTRIBUTE, THRESHOLD);

		} else {
			Attribute best = null;
			double maxGain = 0.0;
			double gain;
			for (Attribute attr : attributeList) {
				gain = this.order(dataExample, attr);
				if (maxGain < gain) {
					maxGain = gain;
					best = attr;
				}
			}

			if (maxGain == 0) {
				int label = Helpers.value(dataExample);
				return new DecTreeNode(label, ATTRIBUTE, THRESHOLD);
			}

			List<List<Integer>> left = Helpers.getLess(best, dataExample);
			List<List<Integer>> right = Helpers.getGreater(best, dataExample);
			DecTreeNode curNode = new DecTreeNode(LABEL, best.getAttr(), best.getThres());
			curNode.left = this.buildTree(left, dataExample, attributeList, depth + 1);
			curNode.right = this.buildTree(right, dataExample, attributeList, depth + 1);

			return curNode;
		}
	}

	public int classify(List<Integer> instance) {
		DecTreeNode curNode = this.root;

		while (!curNode.isLeaf()) {
			if (instance.get(curNode.attribute) <= curNode.threshold) {
				curNode = curNode.left;
			} else {
				curNode = curNode.right;
			}
		}

		return curNode.classLabel;
	}

	public void printTree() {
		printTreeNode("", this.root);
	}

	public void printTreeNode(String prefixStr, DecTreeNode node) {
		String printStr = prefixStr + "X_" + node.attribute;
		System.out.print(printStr + " <= " + String.format("%d", node.threshold));
		if (node.left.isLeaf()) {
			System.out.println(" : " + String.valueOf(node.left.classLabel));
		} else {
			System.out.println();
			printTreeNode(prefixStr + "|\t", node.left);
		}
		System.out.print(printStr + " > " + String.format("%d", node.threshold));
		if (node.right.isLeaf()) {
			System.out.println(" : " + String.valueOf(node.right.classLabel));
		} else {
			System.out.println();
			printTreeNode(prefixStr + "|\t", node.right);
		}
	}

	public double printTest(List<List<Integer>> testDataSet) {
		double numEqual = 0;
		double numTotal = 0;
		for (int i = 0; i <= testDataSet.size() - 1; i++) {
			int truth = testDataSet.get(i).get(testDataSet.get(i).size() - 1);
			int prediction = classify(testDataSet.get(i));
			System.out.println(prediction);
			if (prediction == truth) {
				numEqual++;
			}
			numTotal++;
		}
		double accuracy = numEqual * 100.0 / numTotal;
		System.out.println(String.format("%.2f", accuracy) + "%");
		return accuracy;
	}

	private double order(List<List<Integer>> dataExample, Attribute attribute) {
		double malignant = Helpers.count(dataExample, MALIGNANT);
		double benign = Helpers.count(dataExample, BENIGN);
		double entropy = this.entropy(malignant, benign);
		double entropies = this.entropies(dataExample, attribute);
		double gain = entropy - entropies;

		attribute.setGain(gain);

		return gain;
	}

	private double entropy(double first, double second) {
		double a;
		double b;
		double total = first + second;
		double firstVal = first / total;
		double secondVal = second / total;

		if (secondVal == 0.0) {
			b = 0.0;
		} else {
			b = (Math.log(secondVal) / Math.log(2)) * secondVal;
		}

		if (firstVal == 0.0) {
			a = 0.0;
		} else {
			a = (Math.log(firstVal) / Math.log(2)) * firstVal;
		}

		double entropy = -(a + b);
		return entropy;
	}

	private double entropies(List<List<Integer>> dataExamples, Attribute attribute) {
		List<List<Integer>> left = Helpers.getLess(attribute, dataExamples);
		List<List<Integer>> right = Helpers.getGreater(attribute, dataExamples);

		double leftMalignant = Helpers.count(left, MALIGNANT);
		double leftBenign = Helpers.count(left, BENIGN);
		double rightMalignant = Helpers.count(right, MALIGNANT);
		double rightBenign = Helpers.count(right, BENIGN);

		double left2 = (double) left.size() / (double) dataExamples.size();
		double leftEntropy = this.entropy(leftMalignant, leftBenign);
		double right2 = (double) right.size() / (double) dataExamples.size();
		double rightEntropy = this.entropy(rightMalignant, rightBenign);
		return (leftEntropy * left2) + (rightEntropy * right2);
	}
}

class Attribute {
	private double infoGain;
	private int thres;
	private int attr;

	public Attribute(int attr, int thres) {
		this.attr = attr;
		this.thres = thres;
		this.infoGain = -1;
	}

	void setGain(double gain) {
		this.infoGain = gain;
	}

	double getGain() {
		return this.infoGain;
	}

	int getThres() {
		return this.thres;
	};

	int getAttr() {
		return this.attr;
	}
}

class Helpers {
	static final int LABEL = 9;
	static final int MALIGNANT = 1;
	static final int BENIGN = 0;

	static boolean checkLabels(List<List<Integer>> dataExamples) {
		int testLabel = Helpers.getLabel(dataExamples.get(0));
		for (List<Integer> example : dataExamples) {
			if (Helpers.getLabel(example) != testLabel) {
				return false;
			}
		}

		return true;
	}

	static List<List<Integer>> getLess(Attribute attribute, List<List<Integer>> dataExamples) {
		int attr = attribute.getAttr();
		int threshold = attribute.getThres();

		List<List<Integer>> split = new ArrayList<>();

		for (List<Integer> example : dataExamples) {
			if (example.get(attr) <= threshold) {
				split.add(example);
			}
		}

		return split;
	}

	static List<List<Integer>> getGreater(Attribute attribute, List<List<Integer>> dataExamples) {
		int attr = attribute.getAttr();
		int threshold = attribute.getThres();

		List<List<Integer>> rightSplit = new ArrayList<>();

		for (List<Integer> example : dataExamples) {
			if (example.get(attr) > threshold) {
				rightSplit.add(example);
			}
		}

		return rightSplit;
	}

	static int value(List<List<Integer>> parentEx) {
		int benignCount = Helpers.count(parentEx, BENIGN);
		int malignantCount = Helpers.count(parentEx, MALIGNANT);

		if (benignCount > malignantCount) {
			return 0;
		} else {
			return 1;
		}
	}

	static int count(List<List<Integer>> dataExamples, int label) {
		int count = 0;
		for (List<Integer> example : dataExamples) {
			if (Helpers.getLabel(example) == label) {
				count++;
			}
		}

		return count;
	}

	static int getLabel(List<Integer> example) {
		return example.get(LABEL);
	}
}
