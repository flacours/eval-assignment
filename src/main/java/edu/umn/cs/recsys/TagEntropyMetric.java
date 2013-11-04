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
import java.util.*;

/**
 * A metric that measures the tag entropy of the recommended items.
 * @author <a href="http://www.grouplens.org">GroupLens Research</a>
 */
public class TagEntropyMetric extends AbstractTestUserMetric {
    private final int listSize;
    private final List<String> columns;
    private static Logger logger = LoggerFactory.getLogger(TagEntropyMetric.class);
    private Boolean debug;
    private Boolean debug2;
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
            debug = testUser.getUserId() == 1;
            if(debug) logger.info("TagEntropy evaluate: " + testUser.getUserId());
            debug2 = debug && false;


            double entropy = 0;


            // TODO Implement the entropy metric
            try {

                int l = recommendations.size();

                if(l > 0)
                {

                    MutableSparseVector tagVector = vocab.newTagVector();
                    Map<Long, HashSet<String>> movieTags;
                    movieTags = buildMovieTags(recommendations, tagDAO);
                    double lSum = computeL(movieTags);
                    if(debug) logger.info(String.format("lSum %f", lSum));

                    for(ScoredId scoredId : recommendations)
                    {
                        long movieId = scoredId.getId();
                        HashSet<String> tags = movieTags.get(movieId);
                        for(String tag : tags)
                        {
                            long tagId = vocab.getTagId(tag);
                            if(tagVector.containsKey(tagId) == false)
                            {
                                double px = computePx(movieTags, vocab, tag, lSum);
                                double entropyTag = -px * log2(px);
                               // if(debug) logger.info(String.format("entropy %f - px %f on tag %s", entropyTag, px, tag));
                                //entropy += entropyTag;
                                //if(debug) logger.info(String.format("global entropy %f", entropy));
                                tagVector.set(tagId, entropyTag);
                            }
                        }
                    }
                    entropy = tagVector.sum();
                }
            }
            catch (Exception ex){
                logger.error(String.format("Exception : %s", ex.getMessage()));
                ex.printStackTrace(System.out);
            }
            if(debug) logger.info(String.format("Entropy %f", entropy));
            totalEntropy += entropy;
            userCount += 1;

            return new Object[]{entropy};
        }

        private double computePx( Map<Long, HashSet<String>> movieTags, TagVocabulary vocab, String tag, double lSum)
        {

            if(debug2) logger.info(String.format("compute px on tag %s", tag));
            double sumMovieWithTag = 0;

            for(Long movieId : movieTags.keySet()){
                HashSet<String>tags = movieTags.get(movieId);
                if(tags.contains(tag)) sumMovieWithTag++;
            }
            if(debug2) logger.info(String.format("sum moviewithtag %f on tag %s", sumMovieWithTag, tag));
            return sumMovieWithTag/lSum;
        }

        private double computeL( Map<Long, HashSet<String>> movieTags)
        {
            double lSum = 0;
            for(Long movieId : movieTags.keySet()){
                lSum += movieTags.get(movieId).size();
            }
            return lSum;
        }

        private Map<Long, HashSet<String>> buildMovieTags(List<ScoredId> recommendations, ItemTagDAO tagDAO) {
            Map<Long, HashSet<String>> movieTags = new HashMap<Long, HashSet<String>>();
            for(ScoredId scoredId : recommendations)
            {
                long movieId = scoredId.getId();
                List<String> tags = tagDAO.getItemTags(movieId);
                ArrayList<String> tagsLowerCase = new ArrayList();
                for(String s : tags) {
                    tagsLowerCase.add(s.toLowerCase());
                }
                HashSet<String> uniqueTags = new HashSet<String>();
                for(String tag : tagsLowerCase ) if (!uniqueTags.contains(tag)) uniqueTags.add(tag);
                String v = "";
                for(String s : tagsLowerCase) { v += " " + s; }
                //logger.info("Tags " + v);
                movieTags.put(Long.valueOf(movieId), uniqueTags);
            }

            return  movieTags;
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
