


class mlmDataGenerator:
    def __init__(self):

    def __len__(self):

    def __getitem__(self , idx):

    def mask_tokens(seq):
        '''
        seq : [b x T]

        '''
        T = seq.shape[1]
        num_masks = int(T * 0.15)
        masks_ids = torch.randint(0, T, (num_masks,)) #ids of 15% of the tokens

        #shuffling the ids
        masks_ids=masks_ids[torch.randperm(num_masks)]

        #15% ==> 80% [MASK] , 10% random , 10% keep
        prob = torch.rand(num_masks)
        for i , p in enumerate(prob):
            if p<0.1:
                input{}
            elif 0.1=<p<0.2:
            else :


        random_ratio = int(num_masks*0.1)
        keep_ratio = int(num_masks*0.1)

        random_ids = masks_ids[:random_ratio] #10%
        keep_ids = masks_ids[random_ratio:keep_ratio] #10%
        MASK_ids  = masks_ids[random_ratio+keep_ratio:] #80%

        return masks_ids , MASK_ids , random_ids , keep_ids