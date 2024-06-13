import os
import numpy as np
import torch
from sklearn import metrics
from torch.utils.data import DataLoader


class evaluate():
    def __init__(self, params, graph_classifier, data):
        self.params = params
        self.graph_classifier = graph_classifier
        self.data = data

    def eval(self, save=False):
        dataloader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=False,
                                num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)

        self.graph_classifier.eval()
        num_neg_rel = min(self.params.num_neg_samples_per_link, self.params.num_rels - 1)

        hits1, hits3, hits10, mrr = 0, 0, 0, 0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                data_pos, targets_pos = self.params.move_batch_to_device(batch, self.params.device)
                score_pos, score_neg = self.graph_classifier(data_pos)  # torch.Size([16, 1]) torch.Size([16, 8, 1])

                scores = torch.cat((score_pos, score_neg.view(-1, 1)), dim=0)
                _, indices = torch.sort(scores, descending=True)
                ranks = (indices == 0).nonzero(as_tuple=True)[0] + 1  # 预测正样本的排名

                hits1 += torch.sum(ranks <= 1).item()
                hits3 += torch.sum(ranks <= 3).item()
                hits10 += torch.sum(ranks <= 10).item()
                mrr += torch.sum(1.0 / ranks.float()).item()
                total_samples += len(targets_pos)

        hits1 /= total_samples
        hits3 /= total_samples
        hits10 /= total_samples
        mrr /= total_samples

        if save:
            pos_test_triplets_path = os.path.join(self.params.main_dir,
                                                  'data/{}/{}.txt'.format(self.params.dataset, self.data.file_name))
            with open(pos_test_triplets_path) as f:
                pos_triplets = [line.split() for line in f.read().split('\n')[:-1]]
            pos_file_path = os.path.join(self.params.main_dir,
                                         'data/{}/grail_{}_predictions.txt'.format(self.params.dataset,
                                                                                   self.data.file_name))
            with open(pos_file_path, "w") as f:
                for ([s, r, o], score) in zip(pos_triplets, score_pos.squeeze(1).detach().cpu().tolist()):
                    f.write('\t'.join([s, r, o, str(score)]) + '\n')

            neg_test_triplets_path = os.path.join(self.params.main_dir,
                                                  'data/{}/neg_{}_0.txt'.format(self.params.dataset,
                                                                                self.data.file_name))
            with open(neg_test_triplets_path) as f:
                neg_triplets = [line.split() for line in f.read().split('\n')[:-1]]
            neg_file_path = os.path.join(self.params.main_dir,
                                         'data/{}/grail_neg_{}_{}_predictions.txt'.format(self.params.dataset,
                                                                                          self.data.file_name,
                                                                                          self.params.constrained_neg_prob))
            with open(neg_file_path, "w") as f:
                for ([s, r, o], score) in zip(neg_triplets, score_neg.view(-1, 1).squeeze(1).detach().cpu().tolist()):
                    f.write('\t'.join([s, r, o, str(score)]) + '\n')

        return {'hits@1': hits1, 'hits@3': hits3, 'hits@10': hits10, 'mrr': mrr}
