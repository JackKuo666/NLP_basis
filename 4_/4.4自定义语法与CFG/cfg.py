#encoding=utf8

def exec_cmd(cmd):
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        env=ENVIRON)
    out, err = p.communicate()
    return out, err
import nltk,os,jieba
from nltk.tree import Tree
from nltk.draw import TreeWidget
from nltk.draw.tree import TreeView
from nltk.draw.util import CanvasFrame
from nltk.parse import RecursiveDescentParser
class Cfg():
    '''
    '''

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_sample(self):
        print("test_sample")
        # This is a CFG grammar, where:
        # Start Symbol : S
        # Nonterminal : NP,VP,DT,NN,VB
        # Terminal : "I", "a" ,"saw" ,"dog"
        grammar = nltk.grammar.CFG.fromstring("""
            S -> NP VP
            NP -> DT NN | NN
            VP -> VB NP
            DT -> "a"
            NN -> "I" | "dog"
            VB -> "saw"
        """)
        sentence = "I saw a dog".split()
        parser = RecursiveDescentParser(grammar)
        final_tree = parser.parse(sentence)

        for i in final_tree:
            print(i)

    def test_nltk_cfg_qtype(self):
        print("test_nltk_cfg_qtype")
        gfile = os.path.join(
            curdir,
            os.path.pardir,
            "config",
            "grammar.question-type.cfg")
        question_grammar = nltk.data.load('file:%s' % gfile)

        def get_missing_words(grammar, tokens):
            """
            Find list of missing tokens not covered by grammar
            """
            missing = [tok for tok in tokens
                       if not grammar._lexical_index.get(tok)]
            return missing

        sentence = "what is your name"

        sent = sentence.split()
        missing = get_missing_words(question_grammar, sent)
        target = []
        for x in sent:
            if x in missing:
                continue
            target.append(x)

        rd_parser = RecursiveDescentParser(question_grammar)
        result = []
        print("target: ", target)
        for tree in rd_parser.parse(target):
            result.append(x)
            print("Question Type\n", tree)

        if len(result) == 0:
            print("Not Question Type")

    def cfg_en(self):
        print("test_nltk_cfg_en")
        grammar = nltk.CFG.fromstring("""
         S -> NP VP
         VP -> V NP | V NP PP
         V -> "saw" | "ate"
         NP -> "John" | "Mary" | "Bob" | Det N | Det N PP
         Det -> "a" | "an" | "the" | "my"
         N -> "dog" | "cat" | "cookie" | "park"
         PP -> P NP
         P -> "in" | "on" | "by" | "with"
         """)

        sent = "Mary saw Bob".split()

        rd_parser = RecursiveDescentParser(grammar)

        result = []

        for i, tree in enumerate(rd_parser.parse(sent)):
            result.append(tree)

        assert len(result) > 0, " CFG tree parse fail."

        print(result)

    def cfg_zh(self):
       
        grammar = nltk.CFG.fromstring("""
             S -> N VP
             VP -> V NP | V NP | V N
             V -> "尊敬"
             N -> "我们" | "老师" 
             """)

        sent = "我们 尊敬 老师".split()
        rd_parser = RecursiveDescentParser(grammar)

        result = []

        for i, tree in enumerate(rd_parser.parse(sent)):
            result.append(tree)
            print("Tree [%s]: %s" % (i + 1, tree))

        assert len(result) > 0, "Can not recognize CFG tree."
        if len(result) == 1 :
            print("Draw tree with Display ...")
            result[0].draw()
        else:
            print("WARN: Get more then one trees.")

        print(result)

if __name__ == '__main__':
    cfg=Cfg()
    cfg.cfg_en()
    cfg.cfg_zh()