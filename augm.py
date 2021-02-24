
import imgaug.augmenters as iaa

def weather_augment(a_type,img):
    if a_type=='None':
        return img
    if a_type=='cartoon':
        aug = iaa.Cartoon(blur_ksize=3, segmentation_size=0.6,
                  saturation=1.0, edge_prevalence=0.8)
        img=aug.augment_image(img)
        return img
    if a_type=='zoom':
        if random.random() > 0.5:
            aug = iaa.imgcorruptlike.ZoomBlur(severity=1)
            img = aug.augment_image(img)
        else:
            img = img
        return img
    if a_type=='fog' :
        coeff=random.randint(1,3)
        aug = iaa.imgcorruptlike.Fog(severity=coeff)
        img = aug.augment_image(img)
        return img
    if a_type=='frost':
        coeff=random.randint(1,2)
        aug = iaa.imgcorruptlike.Frost(severity=coeff)
        img = aug.augment_image(img)
        return img
    if a_type=='simple_snow':
        coeff=random.randint(1,3)
        aug = iaa.imgcorruptlike.Snow(severity=coeff)
        img = aug.augment_image(img)
        return img
    ##if a_type=='rain':
    ##    r_type=random.choice(['drizzle','heavy','torrential'])
    ##    img=am.add_rain(img,rain_type=r_type)
    ##    return img
    if a_type=='clouds':
        aug = iaa.Clouds()
        img = aug.augment_image(img)
        return img
