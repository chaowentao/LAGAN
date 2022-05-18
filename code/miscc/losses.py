import torch
import torch.nn as nn
import dlib
import numpy as np
from miscc.config import cfg

from attention import func_attention
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable


# ##################Loss for matching text-image###################
def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def sent_loss(cnn_code, rnn_code, labels, class_ids, batch_size, eps=1e-8):

    masks = []
    if class_ids is not None:
        for i in range(batch_size):
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        masks = np.concatenate(masks, 0)
        masks = torch.ByteTensor(masks)
        if cfg.CUDA:
            masks = masks.cuda()

    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)

    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)

    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * cfg.TRAIN.SMOOTH.GAMMA3

    # --> batch_size x batch_size
    scores0 = scores0.squeeze()
    if class_ids is not None:
        scores0.data.masked_fill_(masks, -float('inf'))
    scores1 = scores0.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(scores0, labels)
        loss1 = nn.CrossEntropyLoss()(scores1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1


def words_loss(img_features, words_emb, labels, cap_lens, class_ids,
               batch_size):
    """
        words_emb(query): batch x nef x seq_len
        img_features(context): batch x nef x 17 x 17
    """
    masks = []
    att_maps = []
    similarities = []
    cap_lens = cap_lens.data.tolist()
    for i in range(batch_size):
        if class_ids is not None:
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        # Get the i-th text description
        words_num = cap_lens[i]
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()
        word = word.repeat(batch_size, 1, 1)
        context = img_features

        weiContext, attn = func_attention(word, context,
                                          cfg.TRAIN.SMOOTH.GAMMA1)

        att_maps.append(attn[i].unsqueeze(0).contiguous())
        word = word.transpose(1, 2).contiguous()
        weiContext = weiContext.transpose(1, 2).contiguous()

        word = word.view(batch_size * words_num, -1)
        weiContext = weiContext.view(batch_size * words_num, -1)
        #
        row_sim = cosine_similarity(word, weiContext)
        row_sim = row_sim.view(batch_size, words_num)

        row_sim.mul_(cfg.TRAIN.SMOOTH.GAMMA2).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True)
        row_sim = torch.log(row_sim)

        similarities.append(row_sim)

    similarities = torch.cat(similarities, 1)
    if class_ids is not None:
        masks = np.concatenate(masks, 0)
        masks = torch.ByteTensor(masks)
        if cfg.CUDA:
            masks = masks.cuda()

    similarities = similarities * cfg.TRAIN.SMOOTH.GAMMA3

    if class_ids is not None:
        similarities.data.masked_fill_(masks, -float('inf'))
    similarities1 = similarities.transpose(0, 1)

    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(similarities, labels)
        loss1 = nn.CrossEntropyLoss()(similarities1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1, att_maps


def mutual_cos_sim(lmks_a, lmks_b):

    a = np.mat(lmks_a)
    b = np.mat(lmks_b)
    # 1. mean
    m_a = np.mean(a, axis=0)
    m_b = np.mean(b, axis=0)
    mean_a = a - m_a
    mean_b = b - m_b
    # 2. norm
    norm_dist_a = np.linalg.norm(mean_a)
    mean_norm_a = mean_a / norm_dist_a
    norm_dist_b = np.linalg.norm(mean_b)
    mean_norm_b = mean_b / norm_dist_b
    # 3. cal trace
    mat_a_b = mean_norm_a.T * mean_norm_b
    ev, _ = np.linalg.eig(mat_a_b)
    # print(ev)
    trace = np.abs((np.sum(ev)))
    # print(trace)
    # 4. arcos
    sim = 1 - np.arccos(min(trace, 1))
    return sim


def lmk_mutual_self_loss(detector, predictor, real_imgs, fake_imgs, captions,
                         batch_size):
    self_loss = 0.
    mutual_loss = 0.
    for i in range(batch_size):
        fake_img = fake_imgs[i]  # [3,256,256]
        fake_img = fake_img.add(1).div(2).mul(255).clamp(0, 255).byte()
        # range from [0, 1] to [0, 255]
        fake_img = fake_img.permute(1, 2, 0).data.cpu().detach().numpy()
        # print(np.shape(fake_img))
        fake_img_flip = np.fliplr(fake_img)  # 水平翻转
        fake_dets = detector(fake_img, 1)
        fake_lmk = []
        for face in fake_dets:
            shape = predictor(fake_img, face)
            for idx, pt in enumerate(shape.parts()):
                pt_pos = (pt.x, pt.y)
                fake_lmk.append(pt_pos)

        fake_flip_dets = detector(fake_img_flip, 1)
        # fake_flip_dets = fake_dets
        fake_flip_lmk = []
        for face in fake_flip_dets:
            shape = predictor(fake_img_flip, face)
            for idx, pt in enumerate(shape.parts()):
                pt_pos = (pt.x, pt.y)
                fake_flip_lmk.append(pt_pos)
        # MSE loss
        if len(fake_dets) == 1 and len(fake_flip_dets) == 1:
            fake_lmk = np.reshape(fake_lmk, (-1, 2))
            fake_flip_lmk = np.reshape(fake_flip_lmk, (-1, 2))
            self_loss += 1. - mutual_cos_sim(fake_lmk, fake_flip_lmk)

        real_img = real_imgs[i]  # [3,256,256]
        real_img = real_img.add(1).div(2).mul(255).clamp(0, 255).byte()
        # range from [0, 1] to [0, 255]
        real_img = real_img.permute(1, 2, 0).data.cpu().detach().numpy()
        # print(np.shape(fake_img))
        real_dets = detector(real_img, 1)
        real_lmk = []
        for face in real_dets:
            shape = predictor(real_img, face)
            for idx, pt in enumerate(shape.parts()):
                pt_pos = (pt.x, pt.y)
                real_lmk.append(pt_pos)
        # mutual cosine loss
        if len(fake_dets) == 1 and len(real_dets) == 1:
            real_lmk = np.reshape(real_lmk, (-1, 2))
            fake_lmk = np.reshape(fake_lmk, (-1, 2))

            cap = captions[i]
            FACE_POINTS = list(range(0, 17))
            BROW_POINTS = list(range(17, 27))
            NOSE_POINTS = list(range(27, 36))
            EYE_POINTS = list(range(36, 48))
            MOUTH_POINTS = list(range(48, 61))
            # face idx in text is 6: attribute face hair
            face_wordtoix = 6
            # eyes idx in text is 14: attribute eyes eyebrows
            eyes_wordtoix = 14
            # mouth idx in text is 21: attribute mouth nose
            mouth_wordtoix = 21
            if face_wordtoix in cap:
                real_lmks_face = real_lmk[FACE_POINTS]
                fake_lmks_face = fake_lmk[FACE_POINTS]
                mutual_loss += 1. - \
                    mutual_cos_sim(real_lmks_face, fake_lmks_face)

            elif eyes_wordtoix in cap:
                real_lmks_eyes_brows = real_lmk[EYE_POINTS + BROW_POINTS]
                fake_lmks_eyes_brows = fake_lmk[EYE_POINTS + BROW_POINTS]
                mutual_loss += 1. - mutual_cos_sim(real_lmks_eyes_brows,
                                                   fake_lmks_eyes_brows)
            elif mouth_wordtoix in cap:
                real_lmks_nose_mouth = real_lmk[NOSE_POINTS + MOUTH_POINTS]
                fake_lmks_nose_mouth = fake_lmk[NOSE_POINTS + MOUTH_POINTS]
                mutual_loss += 1. - mutual_cos_sim(real_lmks_nose_mouth,
                                                   fake_lmks_nose_mouth)

    return self_loss / batch_size, mutual_loss / batch_size


# only one run detector once


def lmk_mutual_self_loss_v2(detector, predictor, real_imgs, fake_imgs,
                            captions, batch_size):
    self_loss = 0.
    mutual_loss = 0.
    for i in range(batch_size):
        fake_img = fake_imgs[i]  # [3,256,256]
        fake_img = fake_img.add(1).div(2).mul(255).clamp(0, 255).byte()
        # range from [0, 1] to [0, 255]
        fake_img = fake_img.permute(1, 2, 0).data.cpu().detach().numpy()
        # print(np.shape(fake_img))
        fake_img_flip = np.fliplr(fake_img)  # 水平翻转
        # fake_dets = detector(fake_img, 1)
        fake_dets = [dlib.rectangle(35, 57, 221, 263)]
        fake_lmk = []
        for face in fake_dets:
            shape = predictor(fake_img, face)
            for idx, pt in enumerate(shape.parts()):
                pt_pos = (pt.x, pt.y)
                fake_lmk.append(pt_pos)

        # fake_flip_dets = detector(fake_img_flip, 1)
        fake_flip_dets = fake_dets
        fake_flip_lmk = []
        for face in fake_flip_dets:
            shape = predictor(fake_img_flip, face)
            for idx, pt in enumerate(shape.parts()):
                pt_pos = (pt.x, pt.y)
                fake_flip_lmk.append(pt_pos)
        # mutual cosine loss
        if len(fake_dets) == 1 and len(fake_flip_dets) == 1:
            fake_lmk = np.reshape(fake_lmk, (-1, 2))
            fake_flip_lmk = np.reshape(fake_flip_lmk, (-1, 2))
            self_loss += 1. - mutual_cos_sim(fake_lmk, fake_flip_lmk)

        real_img = real_imgs[i]  # [3,256,256]
        real_img = real_img.add(1).div(2).mul(255).clamp(0, 255).byte()
        # range from [0, 1] to [0, 255]
        real_img = real_img.permute(1, 2, 0).data.cpu().detach().numpy()
        # print(np.shape(fake_img))
        # real_dets = detector(real_img, 1)
        real_dets = fake_dets
        real_lmk = []
        for face in real_dets:
            shape = predictor(real_img, face)
            for idx, pt in enumerate(shape.parts()):
                pt_pos = (pt.x, pt.y)
                real_lmk.append(pt_pos)
        # mutual cosine loss
        if len(fake_dets) == 1 and len(real_dets) == 1:
            real_lmk = np.reshape(real_lmk, (-1, 2))
            fake_lmk = np.reshape(fake_lmk, (-1, 2))

            cap = captions[i]
            FACE_POINTS = list(range(0, 17))
            BROW_POINTS = list(range(17, 27))
            NOSE_POINTS = list(range(27, 36))
            EYE_POINTS = list(range(36, 48))
            MOUTH_POINTS = list(range(48, 61))
            # face idx in text is 6: attribute face hair
            face_wordtoix = 6
            # eyes idx in text is 14: attribute eyes eyebrows
            eyes_wordtoix = 14
            # mouth idx in text is 21: attribute mouth nose
            mouth_wordtoix = 21
            if face_wordtoix in cap:
                real_lmks_face = real_lmk[FACE_POINTS]
                fake_lmks_face = fake_lmk[FACE_POINTS]
                mutual_loss += 1. - \
                    mutual_cos_sim(real_lmks_face, fake_lmks_face)

            elif eyes_wordtoix in cap:
                real_lmks_eyes_brows = real_lmk[EYE_POINTS + BROW_POINTS]
                fake_lmks_eyes_brows = fake_lmk[EYE_POINTS + BROW_POINTS]
                mutual_loss += 1. - mutual_cos_sim(real_lmks_eyes_brows,
                                                   fake_lmks_eyes_brows)
            elif mouth_wordtoix in cap:
                real_lmks_nose_mouth = real_lmk[NOSE_POINTS + MOUTH_POINTS]
                fake_lmks_nose_mouth = fake_lmk[NOSE_POINTS + MOUTH_POINTS]
                mutual_loss += 1. - mutual_cos_sim(real_lmks_nose_mouth,
                                                   fake_lmks_nose_mouth)

    return self_loss / batch_size, mutual_loss / batch_size


# ##################Loss for G and Ds##############################


def discriminator_loss(netD, real_imgs, fake_imgs, conditions, real_labels,
                       fake_labels, words_embs, cap_lens, image_encoder,
                       class_ids, w_words_embs, wrong_caps_len, wrong_cls_id):

    real_features = netD(real_imgs)
    fake_features = netD(fake_imgs.detach())
    #
    cond_real_logits = netD.COND_DNET(real_features, conditions)
    cond_real_errD = nn.BCELoss()(cond_real_logits, real_labels)
    cond_fake_logits = netD.COND_DNET(fake_features, conditions)
    cond_fake_errD = nn.BCELoss()(cond_fake_logits, fake_labels)
    #
    batch_size = real_features.size(0)
    cond_wrong_logits = netD.COND_DNET(real_features[:(batch_size - 1)],
                                       conditions[1:batch_size])
    cond_wrong_errD = nn.BCELoss()(cond_wrong_logits,
                                   fake_labels[1:batch_size])

    if netD.UNCOND_DNET is not None:
        real_logits = netD.UNCOND_DNET(real_features)
        fake_logits = netD.UNCOND_DNET(fake_features)
        real_errD = nn.BCELoss()(real_logits, real_labels)
        fake_errD = nn.BCELoss()(fake_logits, fake_labels)
        errD = ((real_errD + cond_real_errD) / 2. +
                (fake_errD + cond_fake_errD + cond_wrong_errD) / 3.)
    else:
        errD = cond_real_errD + (cond_fake_errD + cond_wrong_errD) / 2.

    region_features, cnn_code = image_encoder(real_imgs)

    matched = word_level_correlation(region_features, words_embs, cap_lens,
                                     batch_size, class_ids, real_labels)

    mismatched = word_level_correlation(region_features, w_words_embs,
                                        wrong_caps_len, batch_size,
                                        wrong_cls_id, fake_labels)

    errD += (matched + mismatched) / 2.

    return errD


def generator_loss(netsD, image_encoder, fake_imgs, real_labels, words_embs,
                   sent_emb, match_labels, cap_lens, class_ids, style_loss,
                   real_imgs):
    numDs = len(netsD)
    batch_size = real_labels.size(0)
    logs = ''
    errG_total = 0
    perceptual_loss = 0
    ## numDs: 3
    for i in range(numDs):
        features = netsD[i](fake_imgs[i])
        cond_logits = netsD[i].COND_DNET(features, sent_emb)
        cond_errG = nn.BCELoss()(cond_logits, real_labels)
        if netsD[i].UNCOND_DNET is not None:
            logits = netsD[i].UNCOND_DNET(features)
            errG = nn.BCELoss()(logits, real_labels)
            g_loss = errG + cond_errG
        else:
            g_loss = cond_errG
        errG_total += g_loss
        logs += 'g_loss%d: %.2f ' % (i, g_loss)

        if i == (numDs - 1):
            region_features, cnn_code = image_encoder(fake_imgs[i])
            w_loss0, w_loss1, _ = words_loss(region_features, words_embs,
                                             match_labels, cap_lens, class_ids,
                                             batch_size)
            w_loss = (w_loss0 + w_loss1) * \
                cfg.TRAIN.SMOOTH.LAMBDA

            s_loss0, s_loss1 = sent_loss(cnn_code, sent_emb, match_labels,
                                         class_ids, batch_size)
            s_loss = (s_loss0 + s_loss1) * \
                cfg.TRAIN.SMOOTH.LAMBDA

            errG_total += w_loss + s_loss
            logs += 'w_loss: %.2f s_loss: %.2f ' % (w_loss, s_loss)

        fake_img = fake_imgs[i]
        real_img = real_imgs[i]

        real_features = style_loss(real_img)[0]
        fake_features = style_loss(fake_img)[0]
        perceptual_loss += F.mse_loss(real_features,
                                      fake_features) * cfg.TRAIN.SMOOTH.LAMBDA2

    logs += 'perceptual_loss: %.2f ' % (perceptual_loss / 3.)
    errG_total += perceptual_loss / 3.

    return errG_total, logs


def generator_vgg_lmk_mutual_self_loss_v2(netsD, image_encoder, fake_imgs,
                                          real_labels, words_embs, sent_emb,
                                          match_labels, cap_lens, class_ids,
                                          captions, style_loss, detector,
                                          predictor, real_imgs):

    numDs = len(netsD)
    batch_size = real_labels.size(0)
    logs = ''
    # Forward
    errG_total = 0
    perceptual_loss = 0
    for i in range(numDs):
        features = netsD[i](fake_imgs[i])
        cond_logits = netsD[i].COND_DNET(features, sent_emb)
        cond_errG = nn.BCELoss()(cond_logits, real_labels)
        if netsD[i].UNCOND_DNET is not None:
            logits = netsD[i].UNCOND_DNET(features)
            errG = nn.BCELoss()(logits, real_labels)
            g_loss = errG + cond_errG
        else:
            g_loss = cond_errG
        errG_total += g_loss
        # err_img = errG_total
        logs += 'g_loss%d: %.2f ' % (i, g_loss)

        # Ranking loss
        if i == (numDs - 1):
            # words_features: batch_size x nef x 17 x 17
            # sent_code: batch_size x nef
            region_features, cnn_code = image_encoder(fake_imgs[i])
            w_loss0, w_loss1, _ = words_loss(region_features, words_embs,
                                             match_labels, cap_lens, class_ids,
                                             batch_size)
            w_loss = (w_loss0 + w_loss1) * \
                cfg.TRAIN.SMOOTH.LAMBDA
            # err_words = err_words + w_loss

            s_loss0, s_loss1 = sent_loss(cnn_code, sent_emb, match_labels,
                                         class_ids, batch_size)
            s_loss = (s_loss0 + s_loss1) * \
                cfg.TRAIN.SMOOTH.LAMBDA
            # err_sent = err_sent + s_loss

            errG_total += w_loss + s_loss
            logs += 'w_loss: %.2f s_loss: %.2f ' % (w_loss, s_loss)

            fake_img = fake_imgs[i]
            real_img = real_imgs[i]
            self_loss, mutual_loss = lmk_mutual_self_loss_v2(
                detector, predictor, real_img, fake_img, captions, batch_size)

            self_loss = self_loss * cfg.TRAIN.SMOOTH.LAMBDA4
            errG_total += self_loss
            logs += 'self_loss: %.2f ' % (self_loss)
            mutual_loss = mutual_loss * cfg.TRAIN.SMOOTH.LAMBDA3
            errG_total += mutual_loss
            logs += 'mutual_loss: %.2f ' % (mutual_loss)

        fake_img = fake_imgs[i]
        real_img = real_imgs[i]
        real_features = style_loss(real_img)[0]
        fake_features = style_loss(fake_img)[0]
        perceptual_loss += F.mse_loss(real_features,
                                      fake_features) * cfg.TRAIN.SMOOTH.LAMBDA2

    logs += 'perceptual_loss: %.2f ' % (perceptual_loss / 3.)
    errG_total += perceptual_loss / 3.

    return errG_total, logs


##################################################################


##################################################################
def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD


##################################################################


def word_level_correlation(img_features, words_emb, cap_lens, batch_size,
                           class_ids, labels):

    masks = []
    att_maps = []
    result = 0
    cap_lens = cap_lens.data.tolist()
    similar_list = []
    for i in range(batch_size):
        if class_ids is not None:
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))

        words_num = cap_lens[i]

        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()
        context = img_features[i, :, :, :].unsqueeze(0).contiguous()

        weiContext, attn = func_attention(word, context,
                                          cfg.TRAIN.SMOOTH.GAMMA1)

        aver = torch.mean(word, 2)
        averT = aver.unsqueeze(1)
        res_word = torch.bmm(averT, word)
        res_softmax = F.softmax(res_word, 2)
        res_softmax = res_softmax.repeat(1, weiContext.size(1), 1)
        self_weiContext = weiContext * res_softmax

        word = word.transpose(1, 2).contiguous()
        self_weiContext = self_weiContext.transpose(1, 2).contiguous()
        word = word.view(words_num, -1)
        self_weiContext = self_weiContext.view(words_num, -1)
        #
        row_sim = cosine_similarity(word, self_weiContext)
        row_sim = row_sim.view(1, words_num)

        row_sim.mul_(cfg.TRAIN.SMOOTH.GAMMA2).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True)
        row_sim = torch.log(row_sim)
        similar_list.append(F.sigmoid(row_sim[0, 0]))

    similar_list = torch.tensor(similar_list, requires_grad=False).cuda()
    result = nn.BCELoss()(similar_list, labels)

    return result
