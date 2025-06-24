# -*- coding: utf-8 -*-
"""
This module allows to normalize RNAseq gene expression data.
"""
import pandas as pd
import numpy as np
try:
    from rpy2.robjects.packages import importr
    import rpy2.robjects as ro
    from rpy2.robjects import Formula
    from rpy2.robjects import pandas2ri
    deseq2 = importr('DESeq2')
    base = importr('base')
    biogeneric = importr('BiocGenerics')
    sum_exp = importr('SummarizedExperiment')
    pandas2ri.activate()

except:
    print("Warning: could not import R components. DEseq2() function cannot be called.")



def RPM(raw_counts):
    """
    Reads Per Million.

    Args:
        raw_counts (pandas.DataFrame): raw RNAseq counts where rows are genes
            and columns are conditions

    Returns:
        pandas.DataFrame: Normalized counts

    Examples:
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> import pandas as pd
        >>> nb_genes = 1000
        >>> nb_conditions = 5
        >>> raw_counts = np.random.randint(0,1e6,(nb_genes,nb_conditions))
        >>> raw_counts = pd.DataFrame(raw_counts)
        >>> rpm = RPM(raw_counts)
        >>> rpm.head()
                    0            1            2            3            4
        0  1994.031850   617.999738   895.038590   234.467249  1929.409246
        1   308.104674  1783.727269   738.867008   604.569366   245.491264
        2  1235.090833   906.128463   769.221953  1462.698984  1474.653899
        3   628.576824   344.838319  1723.115991  1201.585019  1084.225878
        4  1921.133736  1363.195297   761.440687  1481.020714  1777.679267

    """
    # per million factor: nb of reads per condition divided by 10^6
    pmf = raw_counts.sum(axis=0)/1e6
    rpm = raw_counts / pmf
    return(rpm)

def RPK(raw_counts, seq_lengths, seq_in_kb=False):
    """
    Reads Per Kilobase normalization.

    Args:
        raw_counts (pandas.DataFrame): raw RNAseq counts where rows are genes
            and columns are conditions
        seq_lengths (pandas.Series): sequences DNA lengths
        seq_in_kb (bool): True if lengths in kb, False otherwise

    Returns:
        pandas.DataFrame: Normalized counts

    Examples:
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> import pandas as pd
        >>> nb_genes = 1000
        >>> nb_conditions = 5
        >>> raw_counts = np.random.randint(0,1e6,(nb_genes,nb_conditions))
        >>> raw_counts = pd.DataFrame(raw_counts)
        >>> seq_lengths = np.random.randint(100,20000,nb_genes)
        >>> seq_lengths = pd.Series(seq_lengths)
        >>> rpk = RPK(raw_counts, seq_lengths)
        >>> rpk.head()
                       0              1              2              3              4
        0  321202.997719   99612.577387  142010.101010   38433.365917  313911.697621
        1   26853.843441  155566.114245   63431.417489   53620.768688   21611.248237
        2   97195.319962   71353.390640   59624.960204  117133.237822  117212.034384
        3  132006.796941   72465.590484  356436.703483  256785.896347  229981.733220
        4   48384.227419   34354.424576   18889.143614   37956.492944   45220.490091

    """
    seq_lengths = seq_lengths[raw_counts.index]
    seq_lengths = seq_lengths.fillna(seq_lengths.mean())
    if not seq_in_kb:
        seq_lengths_in_kb = seq_lengths / 1e3
    else:
        seq_lengths_in_kb = seq_lengths
    rpk = (raw_counts.T / seq_lengths_in_kb).T
    return(rpk)

def RPKM(raw_counts, seq_lengths, seq_in_kb=False):
    """
    Reads Per Kilobase Million (also known as FPM: Fragments per kilobase).

    Args:
        raw_counts (pandas.DataFrame): raw RNAseq counts where rows are genes
            and columns are conditions
        seq_lengths (pandas.Series): sequences DNA lengths
        seq_in_kb (bool): True if lengths in kb, False otherwise

    Returns:
        pandas.DataFrame: Normalized counts

    Examples:
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> import pandas as pd
        >>> nb_genes = 1000
        >>> nb_conditions = 5
        >>> raw_counts = np.random.randint(0,1e6,(nb_genes,nb_conditions))
        >>> raw_counts = pd.DataFrame(raw_counts)
        >>> seq_lengths = np.random.randint(100,20000,nb_genes)
        >>> seq_lengths = pd.Series(seq_lengths)
        >>> rpkm = RPKM(raw_counts, seq_lengths)
        >>> rpkm.head()
                    0           1           2           3           4
        0  649.733415  201.368439  291.638511   76.398582  628.676848
        1   54.320288  314.479420  130.265692  106.588393   43.281252
        2  196.607901  144.242035  122.448576  232.839698  234.742741
        3  267.024989  146.490365  731.994898  510.443933  460.588733
        4   97.872216   69.448026   38.791619   75.450645   90.563924

    """
    # reads per million
    rpm = RPM(raw_counts=raw_counts)
    rpkm = RPK(rpm, seq_lengths, seq_in_kb=False)
    return(rpkm)

def TPM(raw_counts, seq_lengths, seq_in_kb=False):
    """
    Transcript Per Million normalization.

    Args:
        raw_counts (pandas.DataFrame): raw RNAseq counts where rows are genes
            and columns are conditions
        seq_lengths (pandas.Series): sequences DNA lengths
        seq_in_kb (bool): True if lengths in kb, False otherwise

    Returns:
        pandas.DataFrame: Normalized counts

    Examples:
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> import pandas as pd
        >>> nb_genes = 1000
        >>> nb_conditions = 5
        >>> raw_counts = np.random.randint(0,1e6,(nb_genes,nb_conditions))
        >>> raw_counts = pd.DataFrame(raw_counts)
        >>> seq_lengths = np.random.randint(100,20000,nb_genes)
        >>> seq_lengths = pd.Series(seq_lengths)
        >>> tpm = TPM(raw_counts, seq_lengths)
        >>> tpm.head()
                     0            1            2            3            4
        0  2455.468465   739.530213  1103.147117   265.510632  2397.256398
        1   205.286894  1154.932887   492.740902   370.430324   165.039097
        2   743.019352   529.732184   463.172003   809.195846   895.115733
        3  1009.139172   537.989227  2768.832068  1773.963432  1756.306584
        4   369.878069   255.049468   146.732550   262.216233   345.336316

    """
    rpk = RPK(raw_counts, seq_lengths, seq_in_kb)
    tpm = RPM(rpk)
    return(tpm)


def DEseq2(raw_counts,col_data,rlog=True):
    """
    Apply R DEseq2 normalization.

    Args:
        raw_counts (pandas.DataFrame): raw RNAseq counts where rows are genes
            and columns are conditions
        col_data (pandas.DataFrame): Two columns, one corresponding to ids of
            each condition (individuals), and one with the experiment id
            (if many repetitions)

    Returns:
        pandas.DataFrame: Normalized counts

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> raw_counts = pd.DataFrame(np.random.randint(0,1000,(20,10)),
                                      columns = ["Z"+str(i) for i in range(10)])
        >>> col_data = pd.DataFrame([["Z0","1"],
                                     ["Z1","2"],
                                     ["Z2","3"],
                                     ["Z3","4"],
                                     ["Z4","5"],
                                     ["Z5","6"],
                                     ["Z6","7"],
                                     ["Z7","8"],
                                     ["Z8","9"],
                                     ["Z9","10"]
                                     ],columns=["individuals","conditions"])
        >>> raw_counts.columns = col_data["individuals"]
        >>> col_data.index = col_data['individuals']
        >>> DEseq2(raw_counts,col_data,rlog=False)
        individuals          X0          X1     ...              X8          X9
        0            408.025477  382.991634     ...        7.745300  611.474516
        1            165.238388  516.593367     ...      270.224902  596.251084
        2            289.912839  377.510537     ...      727.197585   60.893728
        3            463.502625  627.585575     ...      385.543809  718.884285
        4             59.056319  674.174898     ...      364.029087  243.574911
        5            573.263865  181.561329     ...      129.948918  570.878697
        6            304.229522  314.477925     ...      802.068816   44.824550
        7            537.472156  376.825400     ...       36.144732  373.819828
        8            323.914962  608.401737     ...      748.712307  100.643800
        9            464.695682  294.608949     ...      781.414683  535.357356
        10           559.543710   57.551516     ...      112.737140  822.065324
        11           517.786716  123.324676     ...      618.763389  768.783312
        12           222.505121  584.421939     ...       81.755942  166.612005
        13           361.496256  175.395095     ...      333.047888  515.905193
        14           330.476775  666.638390     ...      779.693506  312.926101
        15           331.073304  653.620785     ...      493.978005  787.389729
        16           437.851901   84.271862     ...      483.650938  347.601696
        17           466.485268   28.090621     ...      750.433484    9.303208
        18           459.326926  210.337087     ...      149.742461  468.543405
        19           221.312064  126.065225     ...      662.653421  435.559302

    """
    if len(col_data.index) != len(raw_counts.columns):
        print('each raw of col_data should correspond to each column of raw_counts ... dataframes shapes mismatches')
        return(1)
    if (col_data.index!=raw_counts.columns).any():
        col_data.index = raw_counts.columns
    dds = deseq2.DESeqDataSetFromMatrix(raw_counts,col_data,Formula('~ conditions'))
    dds = deseq2.DESeq(dds)
    res = deseq2.results(dds)
    if rlog:
        rlog_counts = deseq2.rlog(dds)
        normed_counts = sum_exp.assay(rlog_counts)
    else:
        normed_counts = biogeneric.counts(dds,True)
    normed_counts = pd.DataFrame(normed_counts)
    normed_counts.index = raw_counts.index
    normed_counts.columns = raw_counts.columns
    return(normed_counts)

def log(X,base=10, pseudocount=1):
    """
    Add a pseudocount and apply the log transformation with a given base.

    Args:
        X (pandas.DataFrame or numpy.array): gene expression matrix
        base (float): logarithm base
        pseudocount(float): pseudocount value

    Returns:
        pandas.DataFrame or numpy.array: log transformed gene expression matrix

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> data = pd.DataFrame(np.random.randn(5, 5),
                            index=["c1", "c2", "c3", "c4", "c5"],
                            columns=["gene1", "gene2", "gene3", "gene4", "gene5"])
        >>> pseudocount = -np.min(data.values)+1
        >>> log_data = log(data, pseudocount=pseudocount)
        >>> log_data
               gene1     gene2     gene3     gene4     gene5
        c1  0.725670  0.596943  0.656264  0.762970  0.734043
        c2  0.410897  0.653509  0.531687  0.537790  0.598089
        c3  0.567853  0.699600  0.634883  0.565218  0.601718
        c4  0.589577  0.703039  0.524764  0.587268  0.431186
        c5  0.000000  0.623932  0.645169  0.448834  0.765128

    """
    X_log = np.log(X+pseudocount)
    return(X_log/np.log(base))
