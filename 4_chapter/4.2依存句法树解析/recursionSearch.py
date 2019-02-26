#encoding=utf8
import nltk.tree as tree
import nltk
def get_vn_pair():
    pass
def get_noun_chunk(tree):
    noun_chunk=[]
    if tree.label()=="NP":
        nouns_phase=''.join(tree.leaves())
        noun_chunk.append(nouns_phase)   
    return noun_chunk

def get_ip_recursion_noun(tree):
    np_list=[]
    if len(tree)==1:
        tr=tree[0]
        get_ip_recursion_noun(tr)
    if len(tree)==2:
        tr=tree[0]
        get_ip_recursion_noun(tr)        
        tr=tree[1]
        get_ip_recursion_noun(tr)        
    if len(tree)==3:
        tr=tree[0]
        get_ip_recursion_noun(tr)        
        tr=tree[1]
        get_ip_recursion_noun(tr)       
        tr=tree[2]
        get_ip_recursion_noun(tr)    
    if tree.label()=='NP':
        np_list.append(get_noun_chunk(tree))
    return np_list
def get_vv_loss_np(tree):
    if not isinstance(tree,nltk.tree.Tree):
        return False
    stack=[]
    np=[]
    stack.append(tree)
    current_tree=''
    while stack:
        current_tree=stack.pop()
        if isinstance(current_tree,nltk.tree.Tree) and current_tree.label()=='VP':
            continue        
        elif isinstance(current_tree,nltk.tree.Tree) and current_tree.label()!='NP':
            for i in range(len(current_tree)):                
                stack.append(current_tree[i])
        elif isinstance(current_tree,nltk.tree.Tree) and current_tree.label()=='NP':
            np.append(get_noun_chunk(tree))
    if np:
        return np
    else:
        return False
            
def search(tree_in):
    if not isinstance(tree_in,nltk.tree.Tree):
        return False    
    vp_pair=[]  
    stack=[]
    stack.append(tree_in)
    current_tree=''
    while stack:
        tree=stack.pop()
        if isinstance(tree,nltk.tree.Tree) and tree.label()=="ROOT":
            for i in range(len(tree)):
                stack.append(tree[i])	    
        if isinstance(tree,nltk.tree.Tree) and tree.label()=="IP":
            for i in range(len(tree)):
                stack.append(tree[i])	          
        if isinstance(tree,nltk.tree.Tree) and tree.label()=="VP":
            duplicate=[]
            if len(tree)>=2:
                for i in range(1,len(tree)):
                    if tree[0].label()=='VV' and tree[i].label()=="NP":
                        verb=''.join(tree[0].leaves())
                        noun=get_noun_chunk(tree[i])
                        if verb and noun:
                            vp_pair.append((verb,noun))
                            duplicate.append(noun)
                    elif tree[0].label()=='VV' and tree[i].label()!="NP":
                        noun=get_vv_loss_np(tree)
                        verb=''.join(tree[0].leaves())
                        if verb and noun and noun not in duplicate:
                            duplicate.append(noun)
                            vp_pair.append((verb,noun))
    if vp_pair:
        return vp_pair
    else:
        return False                        


    #if tree.label()=="NP":
        #nouns_phase=''.join(tree.leaves())
        #noun_chunk.append(nouns_phase)      
