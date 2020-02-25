import numpy as np
import ot
import torch
import transformers as trns
from scipy import linalg
from sklearn.metrics.pairwise import euclidean_distances


class metric_names:
    FBD = "FBD"
    EMBD = "EMBD"


class BertFeature:
    def __init__(self, bert_model_dir, model_name='bert-base-uncased'):
        self.tokenizer = trns.BertTokenizer.from_pretrained(model_name, cache_dir=bert_model_dir)
        self.model = trns.BertModel.from_pretrained(model_name, cache_dir=bert_model_dir)

    def get_features(self, sentences):
        if type(sentences) is not list:
            sentences = [sentences]
        res = []
        for sentence in sentences:
            input_ids = torch.tensor([self.tokenizer.encode(sentence,
                                                            add_special_tokens=True)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
            with torch.no_grad():
                pooler_output = self.model(input_ids)[1]
                res.append(pooler_output)
        return torch.cat(res, 0).numpy()


# from https://github.com/bioinf-jku/TTUR/blob/master/fid.py
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


class FBD:
    def __init__(self, references, model_name, bert_model_dir):
        # inputs must be list of str

        self.model_name = model_name
        self.bert_model_dir = bert_model_dir

        self.bert_feature = BertFeature(bert_model_dir=bert_model_dir, model_name=model_name)

        self.refrence_mu, self.refrence_sigma = self._calculate_statistics(references)

    def _get_features(self, sentences):
        features = self.bert_feature.get_features(sentences)
        return features

    def _calculate_statistics(self, sentences):
        features = self._get_features(sentences)
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def get_score(self, sentences):
        # inputs must be list of str
        mu, sigma = self._calculate_statistics(sentences)
        return calculate_frechet_distance(self.refrence_mu, self.refrence_sigma, mu, sigma)


class EMBD:
    def __init__(self, references, model_name, bert_model_dir):
        # inputs must be list of str
        self.model_name = model_name
        self.bert_model_dir = bert_model_dir

        self.bert_feature = BertFeature(bert_model_dir=bert_model_dir, model_name=model_name)

        self.reference_features = self._get_features(references)  # sample * feature
        assert self.reference_features.shape[0] == len(references)

    def _get_features(self, sentences):
        features = self.bert_feature.get_features(sentences)
        return features

    def get_score(self, sentences):
        # inputs must be list of str
        features = self._get_features(sentences)
        M = ot.dist(self.reference_features, features, metric="sqeuclidean")
        return ot.emd2(a=[], b=[], M=M)


if __name__ == "__main__":
    # Test1:
    bert_feature = BertFeature(model_name="bert-base-uncased", bert_model_dir="/tmp/Bert/")
    res = bert_feature.get_features(["that is very good", "that is good", "that is bad", "that is very bad"])
    print(euclidean_distances(res))

    # Test2:
    references = ["that is very good", "it is great"]
    sentences1 = ["this is nice", "that is good"]
    sentences2 = ["it is bad", "this is very bad"]

    fbd = FBD(references=references, model_name="bert-base-uncased", bert_model_dir="/tmp/Bert/")
    print(fbd.get_score(sentences=sentences1))
    print(fbd.get_score(sentences=sentences2))

    embd = EMBD(references=references, model_name="bert-base-uncased", bert_model_dir="/tmp/Bert/")
    print(embd.get_score(sentences=sentences1))
    print(embd.get_score(sentences=sentences2))
