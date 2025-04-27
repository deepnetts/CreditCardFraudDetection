package com.deepnetts.examples.creditcardfraud;

import deepnetts.data.MLDataItem;
import deepnetts.data.TabularDataSet;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.logging.Logger;
import java.util.stream.IntStream;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.stat.inference.KolmogorovSmirnovTest;
import tech.tablesaw.api.*;
import tech.tablesaw.columns.Column;
import tech.tablesaw.io.csv.CsvReadOptions;
import tech.tablesaw.plotly.Plot;

import tech.tablesaw.plotly.api.BoxPlot;
import tech.tablesaw.plotly.api.VerticalBarPlot;

import tech.tablesaw.plotly.components.Figure;
import tech.tablesaw.plotly.components.Layout;
import tech.tablesaw.plotly.components.Marker;
import tech.tablesaw.plotly.traces.BarTrace;
import tech.tablesaw.plotly.traces.HistogramTrace;

/**
 *  * Razdvojiti analizu i preprocesiranje Da bude inplace ili return da dobije
 * tabelu kao atribut dizajnirati flow - za cc
 *
 * atribut table da ima metode bez parametra table nego da radi na defaultnoj da
 * iam atribut table konstrutore
 *
 *
 *
 */
public class DataPreparation {

    private Table dataSet;   

    private static final Logger LOGGER = Logger.getLogger(DataPreparation.class.getName()); // DeepNetts.class.getName()
    
    // DataSetAnalysys missingValues, outliers, 
    // ovde napraviti klasu koja sadrzi rezultate svih analzia kao atribute
    // i zdodaj metodu za export u deep netts

    public DataPreparation(Table dataSet) {
        this.dataSet = dataSet;
    }

    public Table getData() {
        return dataSet;
    }

    public DataPreparation(String csvFile) {
        CsvReadOptions options = CsvReadOptions.builder(csvFile)
                .header(true)
                .sample(false) // Read the entire file
                .missingValueIndicator("NaN", "", " ", "-", "nan", "NULL")
                .build();

        this.dataSet = Table.read().usingOptions(options);
    }

    public DataPreparation(CsvReadOptions options) {
        this.dataSet = Table.read().usingOptions(options);
    }

 
    public void countMissingValues() {
        int totalRows = dataSet.rowCount();

        System.out.println(">> Overview of missing values per column:");

        for (Column<?> column : dataSet.columns()) {
            int missing = column.countMissing();
            double percent = (missing * 100.0) / totalRows;

            System.out.printf("Column: %s | Missing: %d (%.2f%%)\n", column.name(), missing, percent);

            if (missing == 0) {
                continue;
            }

            // 25 < , use mean for normal dist, median for non nrmal
            
            if (percent >= 25 && percent <= 40) {
                System.out.println("   -> Recommendation: Consider using more sophisticated imputation methods (e.g., KNN, regression, model-based).");
            } else if (percent > 40) {
                System.out.println("   -> WARNING: Missing percentage is too high. Consider dropping the variable/column.");
                // @FIX:umesto System.out koristiti LOGGER svuda
            //    LOGGER.info("   -> WARNING: Missing percentage is too high. Consider dropping the variable/column.");
            }
        }
    }


    public Table handleMissingValues() {
        List<String> columnsToRemove = new ArrayList<>();
        int totalRows = dataSet.rowCount();

        for (Column<?> column : dataSet.columns()) {
            int missing = column.countMissing(); // izbeci da ovo radi ponovo ako je vec uradio gore sa brojanjem
            double percent = (missing * 100.0) / totalRows;

            System.out.printf("Column: %s | Missing: %d (%.2f%%)\n", column.name(), missing, percent);

            if (missing == 0) {
                continue;
            }

            if (percent > 40) { // best practcie, over 35
                System.out.printf(">> Column %s has too many missing values. It may be removed.\n", column.name());
                columnsToRemove.add(column.name());
                continue;
            }

            if (column instanceof StringColumn) {
                StringColumn sc = (StringColumn) column;
                Table freqTable = sc.countByCategory().sortDescendingOn("Count");
                String mostFrequent = freqTable.stringColumn("Category").get(0); // ??? use case dependent, za manji broj nedostajucih vrednsti , da ne radi po defaultu

                sc.set(sc.isMissing(), mostFrequent);
                System.out.printf(">> Categorical column '%s' - missing values replaced with the most frequent: %s\n", column.name(), mostFrequent);
            } else if (column instanceof DoubleColumn) {
                DoubleColumn dc = (DoubleColumn) column;
                DoubleColumn filtered = dc.where(dc.isNotMissing());

                // Konverzija u double[]
                double[] values = filtered.asDoubleArray();
                double mean = filtered.mean();
                double std = filtered.standardDeviation();

                KolmogorovSmirnovTest ksTest = new KolmogorovSmirnovTest();
                NormalDistribution normalDist = new NormalDistribution(mean, std);

                double pValue = ksTest.kolmogorovSmirnovTest(normalDist, values);
                boolean isNormal = pValue > 0.05;

                if (isNormal) {
                    dc.set(dc.isMissing(), mean);
                    System.out.printf(">> Column '%s' is normally distributed (p=%.4f), NaNs replaced with mean: %.2f\n", column.name(), pValue, mean);
                } else {
                    double median = filtered.median();
                    dc.set(dc.isMissing(), median);
                    System.out.printf(">> Column '%s' is not normally distributed (p=%.4f), NaNs replaced with median: %.2f\n", column.name(), pValue, median);
                }
            }
        }

        System.out.println("\nThe following columns may be removed due to a high percentage of missing values:");
        for (String colName : columnsToRemove) {
            System.out.println(" - " + colName);
        }

        return dataSet;
    }
   
    /**
     * replacece missng falues with most frequent value
     * @param columnName 
     */
    public void imputeCategorical(String columnName) {

        Column<?> column = dataSet.column(columnName);

        // Konvertujemo kolonu u StringColumn samo radi pronalaska najčešće vrednosti
        StringColumn asString = column.asStringColumn();

        Table freqTable = asString.countByCategory().sortDescendingOn("Count");
        String mostFrequent = freqTable.stringColumn("Category").get(0);

        // Prolazimo kroz sve redove i zamenjujemo missing vrednosti
        for (int i = 0; i < column.size(); i++) {
            if (column.isMissing().contains(i)) {
                try {
                    if (column instanceof StringColumn) {
                        ((StringColumn) column).set(i, mostFrequent);
                    } else if (column instanceof IntColumn) {
                        int value = Integer.parseInt(mostFrequent);
                        ((IntColumn) column).set(i, value);
                    } else if (column instanceof DoubleColumn) { // ovo i nema smisla za kategorijske i ovo dole
                        double value = Double.parseDouble(mostFrequent);
                        ((DoubleColumn) column).set(i, value);
                    } else if (column instanceof LongColumn) {
                        long value = Long.parseLong(mostFrequent);
                        ((LongColumn) column).set(i, value);
                    } else {
                        System.out.printf(">> [WARNING] Column '%s' has unsupported type: %s\n", column.name(), column.type());
                    }
                } catch (NumberFormatException e) {
                    System.out.printf(">> [ERROR] Failed to parse '%s' for column '%s'\n", mostFrequent, column.name());
                }
            }
        }

        System.out.printf(">> Column '%s' (treated as categorical) - missing values replaced with the most frequent: %s\n",
                column.name(), mostFrequent);
    }

    public void imputeNumeric(String columnName) {

        DoubleColumn column = (DoubleColumn) dataSet.column(columnName);

        DoubleColumn filtered = column.where(column.isNotMissing());
        double[] values = filtered.asDoubleArray();
        double mean = filtered.mean();
        double std = filtered.standardDeviation();

        KolmogorovSmirnovTest ksTest = new KolmogorovSmirnovTest();
        NormalDistribution normalDist = new NormalDistribution(mean, std);
        double pValue = ksTest.kolmogorovSmirnovTest(normalDist, values);
        boolean isNormal = pValue > 0.05;

        if (isNormal) {
            column.set(column.isMissing(), mean);
            System.out.printf(">> Column '%s' is normally distributed (p=%.4f), NaNs replaced with mean: %.2f\n",
                    column.name(), pValue, mean);
        } else {
            double median = filtered.median();
            column.set(column.isMissing(), median);
            System.out.printf(">> Column '%s' is not normally distributed (p=%.4f), NaNs replaced with median: %.2f\n",
                    column.name(), pValue, median);
        }
    }

    public void showBoxPlotsAndOutliers(List<String> columnNames) {
        for (String columnName : columnNames) {
            Column<?> col = dataSet.column(columnName);

            if (col instanceof NumericColumn<?>) {
                NumericColumn<?> numCol = (NumericColumn<?>) col;
                String numericColName = numCol.name();

                double q1 = numCol.quartile1();
                double q3 = numCol.quartile3();
                double iqr = q3 - q1;
                double lowerBound = q1 - 1.5 * iqr;
                double upperBound = q3 + 1.5 * iqr;

                NumericColumn<?> outliers = numCol.where(numCol.isLessThan(lowerBound).or(numCol.isGreaterThan(upperBound)));

                System.out.println("Column: " + numericColName);
                System.out.println("   Total outliers: " + outliers.size());
                if (outliers.size() > 0) {
                    System.out.println("   Outlier values: " + outliers.unique().asList());
                } else {
                    System.out.println("   No outliers detected.");
                }

                // Dummy kolona za boxplot
                String dummyGroupColName = "group_for_" + numericColName;
                StringColumn groupCol = StringColumn.create(dummyGroupColName, dataSet.rowCount());
                groupCol.setMissingTo("All");
                dataSet.addColumns(groupCol);

                // Kreiranje i prikazivanje boxplot-a
                Figure fig = BoxPlot.create("Boxplot - " + numericColName, dataSet, dummyGroupColName, numericColName);
                Plot.show(fig);

                // Uklanjanje dummy kolone
                dataSet.removeColumns(dummyGroupColName);
                System.out.println("--------------------------------------------------");
            } else {
                System.out.println("Column '" + columnName + "' is not numeric and will not be plotted.");
            }
        }
    }

    public void countUniqueValues() {
        // Prolazimo kroz sve kolone u tabeli
        for (Column<?> col : dataSet.columns()) {
            String colName = col.name();

            // Ispisujemo broj jedinstvenih vrednosti
            int uniqueCount = col.unique().size();
            System.out.println("Column: " + colName + " has " + uniqueCount + " unique values.");
        }
    }

    /**
     * Metod za winsorizaciju kolone podataka, koristi IQR za izračunavanje
     * donje i gornje granice. Limit extreme values to reduce the effect of
     * possibly spurious outliers
     *
     * @param data tabela koja sadrži podatke
     * @param columnName naziv kolone koju treba winsorizovati
     */
    public void winsorize(String columnName) {
        // Pronađi kolonu prema nazivu
        Column<?> col = dataSet.column(columnName);

        // Proveri da li je kolona numerička
        if (col instanceof NumericColumn<?>) {
            NumericColumn<?> numCol = (NumericColumn<?>) col;

            // Ako je to konkretno DoubleColumn, koristimo ga direktno
            if (numCol instanceof DoubleColumn) {
                DoubleColumn doubleCol = (DoubleColumn) numCol;

                // Izračunavanje kvartila
                double q1 = doubleCol.quartile1();
                double q3 = doubleCol.quartile3();
                double iqr = q3 - q1;

                // Izračunavanje donje i gornje granice
                double lowerBound = q1 - 1.5 * iqr;
                double upperBound = q3 + 1.5 * iqr;

                // Ako je donja granica negativna, postavimo je na 0 (ili neki minimalni validni broj)
                if (lowerBound < 0) {
                    lowerBound = 0;
                }

                // Prolazimo kroz sve vrednosti u numeričkoj koloni i vršimo winsorizaciju
                for (int i = 0; i < doubleCol.size(); i++) {
                    double value = doubleCol.get(i);
                    if (value < lowerBound) {
                        doubleCol.set(i, lowerBound);
                    } else if (value > upperBound) {
                        doubleCol.set(i, upperBound);
                    }
                }

                System.out.println("Winsorization completed for column: " + columnName);
                System.out.println("Lower bound: " + lowerBound);
                System.out.println("Upper bound: " + upperBound);

            } else {
                System.out.println("Column '" + columnName + "' is not a DoubleColumn and cannot be winsorized.");
            }
        } else {
            System.out.println("Column '" + columnName + "' is not numeric and cannot be winsorized.");
        }
    }

    public void previewRows(int numberOfRows) {
        // stampaj u logger ali i vrati
        System.out.println("First few rows:");
        System.out.println(dataSet.first(numberOfRows));
    }

    public void statistics() {
        System.out.println(dataSet.summary());
    }

    public void columnInfo() {
        System.out.println(dataSet.structure());
    }

    public void plotCategoricalFeatureVsClassTarget(String featureCol, String targetCol) {

        // Grupisanje i brojanje kombinacija
        Table grouped = dataSet.countBy(featureCol, targetCol);
        String countCol = "Count";

        // Konverzija kolona u string, bez obzira na originalni tip
        StringColumn featureStr = grouped.column(featureCol).asStringColumn();
        StringColumn targetStr = grouped.column(targetCol).asStringColumn();

        List<String> featureValues = dataSet.column(featureCol).asStringColumn().unique().asList();
        List<String> targetValues = dataSet.column(targetCol).asStringColumn().unique().asList();

        List<BarTrace> traces = new ArrayList<>();

        for (String targetVal : targetValues) {
            List<String> xValues = new ArrayList<>();
            List<Number> yValues = new ArrayList<>();

            for (String featureVal : featureValues) {
                // Filtriranje po kombinaciji
                Table filtered = grouped.where(
                        featureStr.isEqualTo(featureVal).and(targetStr.isEqualTo(targetVal))
                );

                int count = filtered.isEmpty() ? 0 : filtered.intColumn(countCol).get(0);
                xValues.add(featureVal);
                yValues.add(count);
            }

            // Kreiranje trace-a
            BarTrace trace = BarTrace.builder(
                    StringColumn.create("x", xValues),
                    DoubleColumn.create("y", yValues)
            ).name(targetVal).build();

            traces.add(trace);
        }

        Layout layout = Layout.builder()
                .title("Plot " + targetCol + " againts " + featureCol)
                .build();

        Plot.show(new Figure(layout, traces.toArray(new BarTrace[0])));
    }

    public void plotNumericFeatureVsClassTarget(String featureCol, String targetCol) {
        // Pretpostavljamo da je target numerički (IntColumn)
        IntColumn target = dataSet.intColumn(targetCol);
        DoubleColumn feature = dataSet.doubleColumn(featureCol);

        // Definisanje boja za svaku klasu
        String[] colors = {"#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"};

        // Lista za čuvanje svih histogram trace-ova
        List<HistogramTrace> traces = new ArrayList<>();

        // Iteriraj kroz svaku jedinstvenu vrednost klase u targetCol
        for (int targetClass : target.unique()) {
            Table filtered = dataSet.where(target.isEqualTo(targetClass));

            // Kreiraj histogram trace za svaki targetClass
            HistogramTrace trace = HistogramTrace.builder(filtered.doubleColumn(featureCol))
                    .name("Class " + targetClass)
                    .marker(Marker.builder().color(colors[targetClass % colors.length]).opacity(0.6).build())
                    .build();
            traces.add(trace);
        }

        // Kreiranje Layout objekta bez korišćenja xaxis() i yaxis() metodama
        Layout layout = Layout.builder()
                .title("Distribution of " + featureCol + " by class " + targetCol)
                .build();

        // Kreiraj Figure sa trace-ovima i layout-om
        Figure fig = new Figure(layout, traces.toArray(new HistogramTrace[0]));

        // Prikazivanje plota
        Plot.show(fig);
    }
    
 public void checkClassBalance(String columnName) { 
        Table negativeExamples = dataSet.where(t -> t.intColumn(columnName).isEqualTo(0));
        Table positiveExamples = dataSet.where(t -> t.intColumn(columnName).isEqualTo(1));

        int numPositive = positiveExamples.rowCount();
        int numNegative = dataSet.rowCount() - numPositive;
        
        String[] categories = {"Not Fraud", "Fraud"};
        int[] values = {numNegative, numPositive};
        
        Table barTable = Table.create("Dist bar")
                .addColumns(
                        StringColumn.create("Categories", categories),
                        IntColumn.create("Values", values)
                );
        Plot.show(VerticalBarPlot.create("Distribution of Fraud and Not Fraud Examples", barTable, "Categories", "Values"));

        System.out.println("Positive class rows: "+numPositive);
        System.out.println("Negative class rows: "+numNegative);   
    }
       
    public static TabularDataSet<MLDataItem> createDataSet(Table table) {
        TabularDataSet<MLDataItem> dataSet= new TabularDataSet(29, 1);
        for (int i = 0; i < table.rowCount(); i++) {            
            Row row = table.row(i);
            
            float[] in = new float[29];
            for(int c=0; c<29; c++) {
                in[c] = (float)row.getDouble(c);
            }
            
            float[] targetOut = new float[1];
            targetOut[0] = row.getInt(29);
                        
            MLDataItem tableRow = new TabularDataSet.Item(in, targetOut);
            dataSet.add(tableRow);
        }
        return dataSet;
    }
    
    public static Table createBalancedSample(Table dataTable, String columnName, long rndSeed) {
        Table negative = dataTable.where(t -> t.intColumn(columnName).isEqualTo(0));
        Table positive = dataTable.where(t -> t.intColumn(columnName).isEqualTo(1));
        
        Table balancedSample = Table.create("Balanced CCF", positive.columns());
        balancedSample.append(positive); // get all positive examples
               
        Random random = new Random(rndSeed);
        IntStream rndIdxs = random.ints(positive.rowCount(), 0, negative.rowCount());              
        int[] randomIdx = rndIdxs.toArray();        
                 
        balancedSample.append(negative.rows(randomIdx));
        return balancedSample;
    }    


}
