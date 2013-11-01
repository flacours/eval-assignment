package edu.umn.cs.recsys;

import com.google.common.collect.ImmutableList;
import edu.umn.cs.recsys.dao.ItemTagDAO;
import org.grouplens.lenskit.core.LenskitRecommender;
import org.grouplens.lenskit.eval.algorithm.AlgorithmInstance;
import org.grouplens.lenskit.eval.data.traintest.TTDataSet;
import org.grouplens.lenskit.eval.metrics.AbstractTestUserMetric;
import org.grouplens.lenskit.eval.metrics.TestUserMetricAccumulator;
import org.grouplens.lenskit.eval.metrics.topn.ItemSelectors;
import org.grouplens.lenskit.eval.traintest.TestUser;
import org.grouplens.lenskit.scored.ScoredId;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.VectorEntry;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.util.List;

/**
 * A metric that measures the tag entropy of the recommended items.
 * @author <a href="http://www.grouplens.org">GroupLens Research</a>
 */
public class TagEntropyMetric extends AbstractTestUserMetric {
    private final int listSize;
    private final List<String> columns;
    private static Logger logger = LoggerFactory.getLogger(TagEntropyMetric.class);
    /**
     * Construct a new tag entropy metric.
     * 
     * @param nitems The number of items to request.
     */
    public TagEntropyMetric(int nitems) {
        listSize = nitems;
        // initialize column labels with list length
        columns = ImmutableList.of(String.format("TagEntropy@%d", nitems));
    }

    /**
     * Make a metric accumulator.  Metrics operate with <em>accumulators</em>, which are created
     * for each algorithm and data set.  The accumulator measures each user's output, and
     * accumulates the results into a global statistic for the whole evaluation.
     *
     * @param algorithm The algorithm being tested.
     * @param data The data set being tested with.
     * @return An accumulator for analyzing this algorithm and data set.
     */
    @Override
    public TestUserMetricAccumulator makeAccumulator(AlgorithmInstance algorithm, TTDataSet data) {
        return new TagEntropyAccumulator();
    }

    /**
     * Return the labels for the (global) columns returned by this metric.
     * @return The labels for the global columns.
     */
    @Override
    public List<String> getColumnLabels() {
        return columns;
    }

    /**
     * Return the lables for the per-user columns returned by this metric.
     */
    @Override
    public List<String> getUserColumnLabels() {
        // per-user and global have the same fields, they just differ in aggregation.
        return columns;
    }


    private class TagEntropyAccumulator implements TestUserMetricAccumulator {
        private double totalEntropy = 0;
        private int userCount = 0;

        /**
         * Evaluate a single test user's recommendations or predictions.
         * @param testUser The user's recommendation result.
         * @return The values for the per-user columns.
         */
        @Nonnull
        @Override
        public Object[] evaluate(TestUser testUser) {
            //logger.info("TagEntropy evaluate: " + testUser.getUserId());

            List<ScoredId> recommendations =
                    testUser.getRecommendations(listSize,
                                                ItemSelectors.allItems(),
                                                ItemSelectors.trainingItems());
            if (recommendations == null) {
                return new Object[1];
            }
            LenskitRecommender lkrec = (LenskitRecommender) testUser.getRecommender();
            ItemTagDAO tagDAO = lkrec.get(ItemTagDAO.class);
            TagVocabulary vocab = lkrec.get(TagVocabulary.class);

            double entropy = 0;


            // TODO Implement the entropy metric
            try {

                int totalMovies = tagDAO.getItemIds().size();
                //logger.info(String.format("Number of movies %d", totalMovies));

                MutableSparseVector tagVector = vocab.newTagVector();
                //logger.info(String.format("nb vocab : %d", tagVector.size()));
                for(ScoredId scoredId : recommendations)
                {
                    long movieId = scoredId.getId();
                    List<String> tags = tagDAO.getItemTags(movieId);
                    int numTagForMovie = tags.size();
                    //logger.info(String.format("num tag %d for movie: %d", movieId, numTagForMovie));
                    for(String tag : tagDAO.getItemTags(movieId) ){
                        long tagId = vocab.getTagId(tag);
                        double val = 1.0/numTagForMovie;
                        if(tagVector.containsKey(tagId) == false) {
                            tagVector.set(tagId, val);
                        } else {
                            tagVector.add(tagId, val);
                        }
                    }
                }
                tagVector.multiply(1.0/totalMovies);
                double sum = tagVector.sum();
                //logger.info(String.format("sum %f ", sum));
                entropy = -sum * log2(sum);
                //logger.info(String.format("entropy = %f", entropy));
            }
            catch (Exception ex){
                logger.error(String.format("Exception : %s", ex.getMessage()));
                ex.printStackTrace(System.out);
            }
            totalEntropy += entropy;
            userCount += 1;

            return new Object[]{entropy};
        }

        private double log2(double x) {
            return Math.log(x)/Math.log(2.0d);
        }


        /**
         * Get the final aggregate results.  This is called after all users have been evaluated, and
         * returns the values for the columns in the global output.
         *
         * @return The final, aggregated columns.
         */
        @Nonnull
        @Override
        public Object[] finalResults() {
            // return a single field, the average entropy
            return new Object[]{totalEntropy / userCount};
        }
    }
}
