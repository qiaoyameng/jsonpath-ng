Added
-----

*   Add versioned, atomic JSON patch application via
    ``jsonpath_ng.ext.patch.VersionedDocument``.  Patches are validated on a
    scratch copy first, so a bad base version or illegal path leaves the
    document and every subscription unchanged.  Applied operations report
    concrete add/remove/replace events, including the paths of array elements
    that shifted.
*   Add bounded subscription sessions via
    ``jsonpath_ng.ext.subscription.SubscriptionSession``.  Registered
    expressions maintain their hit sets incrementally, delivering versioned
    notifications that describe hit-set adds, removes, array moves and value
    replacements.  Field, index, wildcard and explicit scalar filter
    expressions (including root-referencing filters) are maintained locally;
    unsupported expressions are explicitly recomputed in full.  The concrete
    dependencies each maintenance pass actually visits are recorded, and
    closing a session releases the document and subscription references.
