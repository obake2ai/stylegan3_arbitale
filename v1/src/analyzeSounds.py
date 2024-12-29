import librosa
import argparse
import moviepy.editor as mpy
import random
#from scipy.misc import toimage
from tqdm import tqdm
from scipy.stats import truncnorm
import numpy as np
import os


def analyzeSong(args):
    #read song
    if args.song:
        song=args.song
        print('\nReading audio \n')
        y, sr = librosa.load(song)
    else:
        raise ValueError("you must enter an audio file name in the --song argument")


    frame_length=args.frame_length

    #set pitch sensitivity
    pitch_sensitivity=(300-args.pitch_sensitivity) * 512 / frame_length

    #set tempo sensitivity
    tempo_sensitivity=args.tempo_sensitivity * frame_length / 512

    #set depth
    depth=args.depth

    #set number of classes
    num_classes=args.num_classes

    #set sort_classes_by_power
    sort_classes_by_power=args.sort_classes_by_power

    #set jitter
    jitter=args.jitter

    #set truncation
    truncation=args.truncation

    #set batch size
    batch_size=args.batch_size

    #set use_previous_classes
    use_previous_vectors=args.use_previous_vectors

    #set use_previous_vectors
    use_previous_classes=args.use_previous_classes

    #set smooth factor
    if args.smooth_factor > 1:
        smooth_factor=int(args.smooth_factor * 512 / frame_length)
    else:
        smooth_factor=args.smooth_factor

    #set duration
    if args.duration:
        seconds=args.duration
        frame_lim=int(np.floor(seconds*22050/frame_length/batch_size))
    else:
        frame_lim=int(np.floor(len(y)/sr*22050/frame_length/batch_size))

    #create spectrogram
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=512,fmax=8000, hop_length=frame_length)

    #get mean power at each time point
    specm=np.mean(spec,axis=0)

    #compute power gradient across time points
    gradm=np.gradient(specm)

    #set max to 1
    gradm=gradm/np.max(gradm)

    #set negative gradient time points to zero
    gradm = gradm.clip(min=0)

    #normalize mean power between 0-1
    specm=(specm-np.min(specm))/np.ptp(specm)

    #create chromagram of pitches X time points
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=frame_length)

    #sort pitches by overall power
    chromasort=np.argsort(np.mean(chroma,axis=1))[::-1]

    classes = list(range(0,12))
    # if args.classes:
    #     classes=args.classes
    #     if len(classes) not in [12,num_classes]:
    #         raise ValueError("The number of classes entered in the --class argument must equal 12 or [num_classes] if specified")
    #
    # elif args.use_previous_classes==1:
    #     cvs=np.load('class_vectors.npy')
    #     classes=list(np.where(cvs[0]>0)[0])
    #
    # else: #select 12 random classes
    #     cls1000=list(range(1000))
    #     random.shuffle(cls1000)
    #     classes=cls1000[:12]

    if sort_classes_by_power==1:

        classes=[classes[s] for s in np.argsort(chromasort[:num_classes])]

    #get new jitters
    def new_jitters(jitter):
        jitters=np.zeros(512)
        for j in range(512):
            if random.uniform(0,1)<0.5:
                jitters[j]=1
            else:
                jitters[j]=1-jitter
        return jitters


    #get new update directions
    def new_update_dir(nv2,update_dir):
        for ni,n in enumerate(nv2):
            if n >= 2*truncation - tempo_sensitivity:
                update_dir[ni] = -1

            elif n < -2*truncation + tempo_sensitivity:
                update_dir[ni] = 1
        return update_dir


    #smooth class vectors
    def smooth(class_vectors,smooth_factor):

        if smooth_factor==1:
            return class_vectors

        class_vectors_terp=[]
        for c in range(int(np.floor(len(class_vectors)/smooth_factor)-1)):
            ci=c*smooth_factor
            cva=np.mean(class_vectors[int(ci):int(ci)+smooth_factor],axis=0)
            cvb=np.mean(class_vectors[int(ci)+smooth_factor:int(ci)+smooth_factor*2],axis=0)

            for j in range(smooth_factor):
                cvc = cva*(1-j/(smooth_factor-1)) + cvb*(j/(smooth_factor-1))
                class_vectors_terp.append(cvc)

        return np.array(class_vectors_terp)


    #normalize class vector between 0-1
    def normalize_cv(cv2):
        min_class_val = min(i for i in cv2 if i != 0)
        for ci,c in enumerate(cv2):
            if c==0:
                cv2[ci]=min_class_val
        cv2=(cv2-min_class_val)/np.ptp(cv2)

        return cv2

    #From pytorch-pretrained-BigGAN
    def truncated_noise_sample(batch_size=1, dim_z=512, truncation=1., seed=None):
        """ Create a truncated noise vector.
            Params:
                batch_size: batch size.
                dim_z: dimension of z
                truncation: truncation value to use
                seed: seed for the random generator
            Output:
                array of shape (batch_size, dim_z)
        """
        # state = None if seed is None else np.random.RandomState(seed)
        state = np.random.RandomState(0) if seed is None else np.random.RandomState(seed+1)
        values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state).astype(np.float32)
        return truncation * values

    #initialize first class vector
    cv1=np.zeros(len(classes))
    for pi,p in enumerate(chromasort[:num_classes]):
        if num_classes < 12:
            cv1[classes[pi]] = chroma[p][np.min([np.where(chrow>0)[0][0] for chrow in chroma])]
        else:
            cv1[classes[p]] = chroma[p][np.min([np.where(chrow>0)[0][0] for chrow in chroma])]

    #initialize first noise vector
    if args.first_vector:
        nv1 = np.load(args.first_vector)
    else:
        nv1 = truncated_noise_sample(truncation=truncation, seed=args.noise_seed)[0]

    #initialize list of class and noise vectors
    class_vectors=[cv1]
    noise_vectors=[nv1]

    #initialize previous vectors (will be used to track the previous frame)
    cvlast=cv1
    nvlast=nv1


    #initialize the direction of noise vector unit updates
    update_dir=np.zeros(512)
    for ni,n in enumerate(nv1):
        if n<0:
            update_dir[ni] = 1
        else:
            update_dir[ni] = -1


    #initialize noise unit update
    update_last=np.zeros(512)


    ########################################
    ########################################
    ########################################
    ########################################
    ########################################

    print('\nGenerating input vectors \n')

    for i in tqdm(range(len(gradm))):

        #print progress
        pass

        #update jitter vector every 100 frames by setting ~half of noise vector units to lower sensitivity
        if i%200==0:
            jitters=new_jitters(jitter)

        #get last noise vector
        nv1=nvlast

        #set noise vector update based on direction, sensitivity, jitter, and combination of overall power and gradient of power
        update = np.array([tempo_sensitivity for k in range(512)]) * (gradm[i]+specm[i]) * update_dir * jitters

        #smooth the update with the previous update (to avoid overly sharp frame transitions)
        update=(update+update_last*3)/4

        #set last update
        update_last=update

        #update noise vector
        nv2=nv1+update

        #append to noise vectors
        noise_vectors.append(nv2)

        #set last noise vector
        nvlast=nv2

        #update the direction of noise units
        update_dir=new_update_dir(nv2,update_dir)

        #get last class vector
        cv1=cvlast

        #generate new class vector
        cv2=np.zeros(len(classes))
        for j in range(num_classes):

            cv2[classes[j]] = (cvlast[classes[j]] + ((chroma[chromasort[j]][i])/(pitch_sensitivity)))/(1+(1/((pitch_sensitivity))))

        #if more than 6 classes, normalize new class vector between 0 and 1, else simply set max class val to 1
        if num_classes > 6:
            cv2=normalize_cv(cv2)
        else:
            cv2=cv2/np.max(cv2)

        #adjust depth
        cv2=cv2*depth

        #this prevents rare bugs where all classes are the same value
        if np.std(cv2[np.where(cv2!=0)]) < 0.0000001:
            cv2[classes[0]]=cv2[classes[0]]+0.01

        #append new class vector
        class_vectors.append(cv2)

        #set last class vector
        cvlast=cv2


    #interpolate between class vectors of bin size [smooth_factor] to smooth frames
    #class_vectors=smooth(class_vectors,smooth_factor)

    expanded_noise_vectors = np.array(noise_vectors)[:, np.newaxis, :]
    for i in range(18-1):
        expanded_noise_vectors = np.concatenate([expanded_noise_vectors, np.array(noise_vectors)[:, np.newaxis, :]], 1)


    #check whether to use vectors from last run
    if use_previous_vectors==1:
        #load vectors from previous run
        class_vectors=np.load('%s_class_vectors.npy')
        noise_vectors=np.load('%s_noise_vectors.npy')
    else:
        #save record of vectors for current video
        # np.save(os.path.join(args.output_path, '%s_class_vectors.npy' %os.path.basename(args.song).split('.')[0]), class_vectors)
        # np.save(os.path.join(args.output_path, '%s_noise_vectors.npy' %os.path.basename(args.song).split('.')[0]), expanded_noise_vectors)
        np.save(os.path.join(args.output_path, 'class_vectors.npy'), class_vectors)
        np.save(os.path.join(args.output_path, 'noise_vectors.npy'), expanded_noise_vectors)
        return expanded_noise_vectors, class_vectors

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--song",required=True)
    parser.add_argument("--resolution", default='512')
    parser.add_argument("--duration", type=int)
    parser.add_argument("--pitch_sensitivity", type=int, default=220)
    parser.add_argument("--tempo_sensitivity", type=float, default=0.25)
    parser.add_argument("--depth", type=float, default=1)
    parser.add_argument("--classes", nargs='+', type=int)
    parser.add_argument("--num_classes", type=int, default=12)
    parser.add_argument("--sort_classes_by_power", type=int, default=0)
    parser.add_argument("--jitter", type=float, default=0.5)
    parser.add_argument("--frame_length", type=int, default=512)
    parser.add_argument("--truncation", type=float, default=1)
    parser.add_argument("--smooth_factor", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=30)
    parser.add_argument("--use_previous_classes", type=int, default=0)
    parser.add_argument("--use_previous_vectors", type=int, default=0)
    parser.add_argument("--output_path", default=".", type=str)
    parser.add_argument("--first_vector", default=None, type=str)
    parser.add_argument("--noise_seed", default=None, type=int)
    args = parser.parse_args()
    v = analyzeSong(args)
