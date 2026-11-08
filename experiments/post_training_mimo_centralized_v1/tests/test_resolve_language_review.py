import unittest
from experiments.post_training_mimo_centralized_v1.scripts import resolve_language_review as r
from experiments.post_training_mimo_centralized_v1.scripts import select_language_replay as s


def candidate(key,cluster,cls="c"):
    origin=s.Origin(key,1,"a"*64,"b"*64,cluster,cls,"communication",None)
    return s.Candidate({"id":key},b'{"id":"'+key.encode()+b'"}\n',1,origin)


class ReviewResolutionTests(unittest.TestCase):
    def setUp(self):
        self.p=candidate("p","one");self.alt=candidate("a","one")
        self.q=candidate("q","two");self.next=candidate("n","three")

    def test_same_cluster_preserves_other_primaries(self):
        out,log=r.resolve([self.p,self.q],[self.alt],[self.p,self.q,self.alt],
             {"p":"FLAG_REVIEW","q":"ACCEPTABLE_REFERENCE"},
             {"p":{"new_id":"a","mode":"same_cluster_ranked_alternative","reason":"source review"}},7)
        self.assertEqual(out,[self.alt,self.q]);self.assertEqual(len(log),1)
        self.assertEqual(out[0].raw_line,self.alt.raw_line)

    def test_unflagged_replacement_refused(self):
        with self.assertRaises(ValueError):r.resolve([self.p],[self.alt],[self.p,self.alt],
             {"p":"ACCEPTABLE_REFERENCE"},{"p":{"new_id":"a","mode":"same_cluster_ranked_alternative"}},7)

    def test_changed_class_or_unranked_candidate_refused(self):
        for new in [candidate("a","one","different"),self.alt]:
            with self.assertRaises(ValueError):r.resolve([self.p],[],[self.p,new],{"p":"FLAG_REVIEW"},
              {"p":{"new_id":"a","mode":"same_cluster_ranked_alternative","reason":"review"}},7)

    def test_next_cluster_follows_frozen_order(self):
        alt=candidate("x","four")
        winner=min([self.next,alt],key=lambda c:(s.cluster_order(7,"c",c.cluster),c.cluster))
        loser=alt if winner==self.next else self.next
        args=([self.p,self.q],[],[self.p,self.q,self.next,alt],{"p":"FLAG_REVIEW","q":"ACCEPTABLE_REFERENCE"})
        rule=lambda x:{"p":{"new_id":x.origin.row_id,"mode":"next_unselected_cluster","reason":"exhausted alternatives"}}
        out,_=r.resolve(*args,rule(winner),7);self.assertEqual(out[0],winner)
        with self.assertRaises(ValueError):r.resolve(*args,rule(loser),7)

    def test_missing_review_refused(self):
        with self.assertRaises(ValueError):r.resolve([self.p,self.q],[],[self.p,self.q],{"p":"ACCEPTABLE_REFERENCE"},{},7)


if __name__=="__main__":unittest.main()
